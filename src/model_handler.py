import os
import torch
import mlflow
import warnings
import numpy as np
from utils.model_supervised import SupervisedSegmentationModel
from Probabilistic_Unet_Pytorch.utils import l2_regularisation
from utils.model_confusionmatrix import Crowd_segmentationModel
from Probabilistic_Unet_Pytorch.probabilistic_unet import ProbabilisticUnet
from utils.model_pionono import PiononoModel
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import utils.globals as globals
from utils.saving import save_model, save_results, save_test_images, save_image_color_legend, save_crowd_images, \
    save_grad_flow, save_test_image_variability, save_model_distributions
from utils.loss import noisy_label_loss
from utils.test_helpers import segmentation_scores
from utils.mlflow_logger import log_results, probabilistic_model_logging, set_epoch_output_dir, set_test_output_dir

eps=1e-7


class ModelHandler():
    def __init__(self, annotators):
        config = globals.config
        self.train_img_vis = []
        self.train_label_vis = []
        self.train_pred_vis = []
        self.train_img_name = []
        self.epoch = 0
        # architecture

        if config['model']['method'] == 'prob-unet':
            self.model = ProbabilisticUnet(input_channels=3, num_classes=config['data']['class_no'],
                                           latent_dim=config['model']['prob_unet_config']['latent_dim'],
                                           no_convs_fcomb=4, beta=config['model']['prob_unet_config']['kl_factor'],
                                           reg_factor=config['model']['prob_unet_config']['reg_factor'],
                                           original_backbone=config['model']['prob_unet_config']['original_backbone'])
            # (256, 128, 64, 32, 16)
            # self.model = ProbabilisticUnet(3, config['data']['class_no'])
        elif config['model']['method'] == 'pionono':
            self.model = PiononoModel(input_channels=3, num_classes=config['data']['class_no'], num_annotators=len(annotators),
                                      predict_annotator=config['model']['pionono_config']['gold_annotator'],
                                      latent_dim=config['model']['pionono_config']['latent_dim'],
                                      z_prior_mu=config['model']['pionono_config']['z_prior_mu'],
                                      z_prior_sigma=config['model']['pionono_config']['z_prior_sigma'],
                                      z_posterior_init_sigma=config['model']['pionono_config']['z_posterior_mu_rand_sigma'],
                                      no_head_layers=config['model']['pionono_config']['no_head_layers'],
                                      kl_factor=config['model']['pionono_config']['kl_factor'],
                                      reg_factor=config['model']['pionono_config']['reg_factor'],
                                      mc_samples=config['model']['pionono_config']['mc_samples']
                                      )
        elif config['model']['method'] == 'confusion_matrix':
            self.model = Crowd_segmentationModel(annotators)
            self.alpha = 1
            self.annotators = annotators
        else:
            self.model = SupervisedSegmentationModel()

        # loss
        self.loss_mode = config['model']['loss']

        #GPU
        self.model.cuda()
        if torch.cuda.is_available():
            print('Running on GPU')
            self.device = torch.device('cuda')
        else:
            warnings.warn("Running on CPU because no GPU was found!")
            self.device = torch.device('cpu')

    def train(self, trainloader, validateloader):
        config = globals.config
        model = self.model
        device = self.device
        max_score = 0
        c_weights = config['data']['class_weights']
        class_weights = torch.FloatTensor(c_weights).cuda()

        class_no = config['data']['class_no']
        epochs = config['model']['epochs']
        learning_rate = config['model']['learning_rate']
        batch_s = config['model']['batch_size']

        save_image_color_legend()

        # Optimizer

        if config['model']['method'] == 'pionono':
            opt_params = [
                {'params': model.unet.parameters()},
                {'params': model.head.parameters()},
                {'params': model.z.parameters(), 'lr': config['model']['pionono_config']['z_learning_rate']}
            ]
        elif config['model']['method'] == 'confusion_matrix':
            opt_params = [
                {'params': model.seg_model.parameters()},
                {'params': model.crowd_layers.parameters(), 'lr': 1e-3}
            ]
        else:
            opt_params = [{'params': model.parameters()}]

        if config['model']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(opt_params, lr=learning_rate)
        elif config['model']['optimizer'] == 'sgd_mom':
            optimizer = torch.optim.SGD(opt_params, lr=learning_rate, momentum=0.9, nesterov=True)
        else:
            raise Exception('Choose valid optimizer!')

        if config['data']['ignore_last_class']:
            ignore_index = int(config['data']['class_no'])  # deleted class is always set to the last index
        else:
            ignore_index = -100  # this means no index ignored
        self.ignore_index = ignore_index
        if config['model']['method'] != 'conf_matrix':
            if self.loss_mode == 'ce':
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index, weight=class_weights)
            elif self.loss_mode == 'dice':
                loss_fct = DiceLoss(ignore_index=ignore_index, from_logits=False, mode='multiclass')
            elif self.loss_mode == 'focal':
                loss_fct = FocalLoss(reduction='mean', ignore_index=ignore_index, mode='multiclass')

        loss_dict = {}
        # Training loop
        for i in range(0, epochs):
            print('\nEpoch: {}'.format(i))
            model.train()
            set_epoch_output_dir(i)
            self.epoch = i

            # Training in batches
            for j, (images, labels, imagename, ann_ids) in enumerate(trainloader):
                # Loading data to GPU
                images = images.cuda().float()
                labels = labels.cuda().long()
                ann_ids = ann_ids.cuda().float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # Foward+loss (crowd or not)
                _, labels = torch.max(labels, dim=1)

                if config['model']['method'] == 'prob-unet':
                    # labels = torch.unsqueeze(labels, 1)
                    model.forward(images, labels, training=True)
                    loss = model.combined_loss(labels, loss_fct)
                    y_pred = model.reconstruction
                elif config['model']['method'] == 'pionono':
                    model.forward(images)
                    loss = model.combined_loss(labels, loss_fct, ann_ids)
                    y_pred = model.preds
                elif config['model']['method'] == 'supervised':
                    y_pred = model(images)
                    loss = loss_fct(y_pred, labels)
                elif config['model']['method'] == 'conf_matrix':
                    if i == 5:  # 10 for cr_image_dice // 5 rest of the methods
                        print("Minimize trace activated!")
                        min_trace = True
                        self.alpha = config['model']['alpha']
                        print("Alpha updated", self.alpha)
                        optimizer = torch.optim.Adam([
                            {'params': model.seg_model.parameters()},
                            {'params': model.crowd_layers.parameters(), 'lr': 1e-4}
                        ], lr=learning_rate)
                    y_pred, cms = model(images, ann_ids)
                    loss, loss_ce, loss_trace = noisy_label_loss(y_pred, cms, labels, ignore_index,
                                                                 config['model']['min_trace'], self.alpha,
                                                                 self.loss_mode)
                else:
                    print('Choose valid model method!')

                if config['model']['method'] != 'conf_matrix':
                    if j % int(config['logging']['interval']) == 0:
                        print("Iter {}/{} - batch loss : {:.4f}".format(j, len(trainloader), loss))
                        self.log_training_metrics(y_pred, labels, loss, model, i * len(trainloader) * batch_s + j)
                    self.store_train_imgs(imagename, images, labels, y_pred)

                # Backprop
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

            # Save validation results
            val_results = self.evaluate(validateloader, mode='val')  # TODO: validate crowd
            log_results(val_results, mode='val', step=int((i + 1) * len(trainloader) * batch_s))
            mlflow.log_metric('finished_epochs', self.epoch + 1, int((i + 1) * len(trainloader) * batch_s))
            # Save model
            metric_for_saving = val_results['macro_dice']
            if max_score < metric_for_saving and i > 10:
                save_model(model)
                max_score = metric_for_saving

            if i % int(config['logging']['artifact_interval']) == 0:
                save_model_distributions(model)
                save_grad_flow(model.named_parameters())
                self.save_train_imgs()

            mlflow.log_artifacts(globals.config['logging']['experiment_folder'])

            # LR decay
            if i > config['model']['lr_decay_after_epoch']:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / (1 + config['model']['lr_decay_param'])

            # Show annotator matrix
            if config['data']['crowd'] and config['model']['method'] == 'conf_matrix':
                _,  ann_id = torch.max(ann_ids, dim=1)
                for ann_ix, cm in enumerate(cms):
                    cm = cm.view(5,5,512,512)
                    cm_ = cm[:,:,100,100]
                    cm_ = cm_/cm_.sum(0)
                    print("Annotators", ann_id)
                    print("CM ", ann_id[ann_ix].cpu().detach().numpy()+1, ": ", cm_.cpu().detach().numpy())


        # Final evaluation of crowd
        if config['data']['crowd'] and config['model']['method']=='conf_matrix':
            self.evaluate_confusion_matrix_model(trainloader, mode='train')

    def test(self, testloader):
        set_test_output_dir()
        save_image_color_legend()
        results = self.evaluate(testloader)
        log_results(results, mode='test', step=None)
        save_results(results)
        mlflow.log_artifacts(globals.config['logging']['experiment_folder'])

    def evaluate(self, evaluatedata, mode='test'):
        config = globals.config
        class_no = config['data']['class_no']
        vis_images = config['data']['visualize_images'][mode]

        if mode=='test':
            print("Testing the best model")
            model_dir = 'models'
            dir = os.path.join(globals.config['logging']['experiment_folder'], model_dir)
            model_path = os.path.join(dir, 'best_model.pth')
            model = torch.load(model_path)
        else:
            model = self.model

        device = self.device
        model.eval()

        labels = []
        preds = []

        with torch.no_grad():
            for j, (test_img, test_label, test_name, ann_id) in enumerate(evaluatedata):
                test_img = test_img.to(device=device, dtype=torch.float32)
                if config['model']['method'] == 'prob-unet':
                    model.forward(test_img, None, training=False)
                    test_pred = model.sample(testing=True)
                elif config['model']['method'] == 'pionono':
                    model.forward(test_img)
                    test_pred = model.sample(use_z_mean=True)
                else:
                    test_pred = model(test_img)
                _, test_pred = torch.max(test_pred[:, 0:class_no], dim=1)
                test_pred_np = test_pred.cpu().detach().numpy()
                test_label = test_label.cpu().detach().numpy()
                test_label = np.argmax(test_label, axis=1)

                preds.append(test_pred_np.astype(np.int8).copy().flatten())
                labels.append(test_label.astype(np.int8).copy().flatten())
                if self.epoch % int(config['logging']['artifact_interval']) == 0 or mode == 'test':
                    for k in range(len(test_name)):
                        if test_name[k] in vis_images or vis_images == 'all':
                            img = test_img[k]
                            save_test_images(img, test_pred_np[k], test_label[k], test_name[k], mode)
                            save_test_image_variability(model, test_name, k, mode)

            preds = np.concatenate(preds, axis=0, dtype=np.int8).flatten()
            labels = np.concatenate(labels, axis=0, dtype=np.int8).flatten()

            results = self.get_results(preds, labels)

            print('RESULTS for ' + mode)
            print(results)
            return results

    def evaluate_confusion_matrix_model(self, evaluatedata, mode='train'):
        config = globals.config
        class_no = config['data']['class_no']
        vis_images = config['data']['visualize_images'][mode]
        print("Testing the best model for crowds")
        model_dir = 'models'
        dir = os.path.join(globals.config['logging']['experiment_folder'], model_dir)
        model_path = os.path.join(dir, 'best_model.pth')
        model = torch.load(model_path)

        device = self.device
        model.eval()

        with torch.no_grad():
            for j, (test_img, test_label, test_name, ann_id) in enumerate(evaluatedata):
                test_img = test_img.to(device=device, dtype=torch.float32)
                ann_id = ann_id.to(device=device)
                pred_noisy_list = []
                test_pred, cm = model(test_img, ann_id)

                test_pred_np = test_pred.cpu().detach().numpy()
                test_pred_np = np.argmax(test_pred_np, axis=1)

                _, test_label = torch.max(test_label, dim=1)
                test_label = test_label.cpu().detach().numpy()

                b, c, h, w = test_pred.size()

                pred_noisy = test_pred.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)

                cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(
                    b * h * w, c, c)
                cm = cm / cm.sum(1, keepdim=True) # normalize cm

                pred_noisy = torch.bmm(cm, pred_noisy).view(b * h * w, c) # prediction annotator
                pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

                _, pred_noisy = torch.max(pred_noisy[:, 0:class_no], dim=1)
                pred_noisy_np = pred_noisy.cpu().detach().numpy()

                pred_noisy_list.append(pred_noisy.cpu().detach().numpy().astype(np.int8).copy().flatten())

                cm = cm.view(b, h*w, c, c)

                config = globals.config
                if config['model']['method'] == 'pixel':
                    cm = cm.mean(1)
                    cm = cm/cm.sum(1, keepdim=True)

                else:
                    cm = cm[:,0,:,:]
                _, ann = torch.max(ann_id, dim=1)
                ann = ann.cpu().detach().numpy()
                for k in range(len(test_name)):
                    if test_name[k] in vis_images or vis_images == 'all':
                        img = test_img[k]
                        save_crowd_images(img, test_pred_np[k], pred_noisy_np[k], test_label[k],
                                          test_name[k], self.annotators[ann[k]], cm[k])

    def evaluate_sr(self, evaluatedata, mode='sr'):
        config = globals.config
        class_no = config['data']['class_no']
        vis_images = config['data']['visualize_images'][mode]
        print("Testing the best model for crowds")
        model_dir = 'models'
        dir = os.path.join(globals.config['logging']['experiment_folder'], model_dir)
        model_path = os.path.join(dir, 'best_model.pth')
        model = torch.load(model_path)

        device = self.device
        model.eval()

        ann_ids = torch.eye(20)

        with torch.no_grad():
            for j, (test_img, test_label, test_name, _) in enumerate(evaluatedata):
                test_img = test_img.to(device=device, dtype=torch.float32)
                ann_id = ann_id.to(device=device)
                pred_noisy_list = []

                for ann_id in ann_ids:
                    test_pred, cm = model(test_img, ann_id)

                    test_pred_np = test_pred.cpu().detach().numpy()
                    test_pred_np = np.argmax(test_pred_np, axis=1)

                    _, test_label = torch.max(test_label, dim=1)
                    test_label = test_label.cpu().detach().numpy()

                    b, c, h, w = test_pred.size()

                    pred_noisy = test_pred.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)

                    cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(
                        b * h * w, c, c)
                    cm = cm / cm.sum(1, keepdim=True) # normalize cm

                    pred_noisy = torch.bmm(cm, pred_noisy).view(b * h * w, c) # prediction annotator
                    pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

                    _, pred_noisy = torch.max(pred_noisy[:, 0:class_no], dim=1)
                    pred_noisy_np = pred_noisy.cpu().detach().numpy()

                    pred_noisy_list.append(pred_noisy.cpu().detach().numpy().astype(np.int8).copy().flatten())

                    cm = cm.view(b, h*w, c, c)

                    config = globals.config
                    if config['model']['method'] == 'pixel':
                        cm = cm.mean(1)
                        cm = cm/cm.sum(1, keepdim=True)

                    else:
                        cm = cm[:,0,:,:]
                    _, ann = torch.max(ann_id, dim=1)
                    ann = ann.cpu().detach().numpy()
                    for k in range(len(test_name)):
                        if test_name[k] in vis_images or vis_images == 'all':
                            img = test_img[k]
                            save_crowd_images(img, test_pred_np[k], pred_noisy_np[k], test_label[k],
                                              test_name[k], self.annotators[ann[k]], cm[k])
    def get_results(self, pred, label):
        config = globals.config
        class_no = config['data']['class_no']
        class_names = globals.config['data']['class_names']

        if globals.config['data']['ignore_last_class_only_for_testing']:
            class_no = class_no-1

        metrics_names = ['macro_dice', 'micro_dice', 'miou', 'accuracy', 'macro_f1', 'cohens_kappa']
        for class_id in range(class_no):
            metrics_names.append('dice_class_' + str(class_id) + '_' + class_names[class_id])
            metrics_names.append('f1_class_' + str(class_id) + '_' + class_names[class_id])

        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy().copy().flatten()
        if torch.is_tensor(label):
            label = label.cpu().detach().numpy().copy().flatten()

        results = segmentation_scores(label, pred, metrics_names)

        return results

    def log_training_metrics(self, y_pred, labels, loss, model, step):
        config = globals.config
        _, y_pred = torch.max(y_pred[:, 0:config['data']['class_no']], dim=1)
        mlflow.log_metric('loss',float(loss.cpu().detach().numpy()), step=step)
        train_results = self.get_results(y_pred, labels)
        log_results(train_results, mode='train', step=step)
        probabilistic_model_logging(model, step)

    def store_train_imgs(self, imagenames, images, labels, y_pred):
        config = globals.config
        vis_train_images = config['data']['visualize_images']['train']

        for k in range(len(imagenames)):
            if imagenames[k] in vis_train_images:
                _, y_pred_argmax = torch.max(y_pred[:, 0:config['data']['class_no']], dim=1)
                self.train_img_vis.append(images[k])
                self.train_label_vis.append(labels[k].cpu().detach().numpy())
                self.train_pred_vis.append(y_pred_argmax[k].cpu().detach().numpy())
                self.train_img_name.append(imagenames[k])

                # if len(y_pred[k].cpu().detach().numpy().shape) == 1:
                #     print(y_pred[k])

    def save_train_imgs(self):
        for i in range(len(self.train_img_vis)):
            save_test_images(self.train_img_vis[i], self.train_pred_vis[i],
                             self.train_label_vis[i], self.train_img_name[i], 'train')
        self.train_img_vis = []
        self.train_label_vis = []
        self.train_pred_vis = []
        self.train_img_name = []