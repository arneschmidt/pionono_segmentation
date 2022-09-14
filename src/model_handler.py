import os
import torch
import mlflow
import warnings
import numpy as np
from Probabilistic_Unet_Pytorch.probabilistic_unet import ProbabilisticUnet
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import utils.globals as globals
from utils.saving import save_model, save_results, save_test_images, save_image_color_legend, save_crowd_images
from utils.model_architecture import SegmentationModel
from Probabilistic_Unet_Pytorch.utils import l2_regularisation
from utils.crowd_model_architecture import Crowd_segmentationModel
from utils.loss import noisy_label_loss
from utils.test_helpers import segmentation_scores
from utils.logging import log_results

eps=1e-7


class ModelHandler():
    def __init__(self, annotators):
        config = globals.config

        # architecture
        if config['model']['crowd_type'] == 'prob-unet':
            self.model = ProbabilisticUnet(3, config['data']['class_no'])
        elif config['data']['crowd']:
            self.model = Crowd_segmentationModel(annotators)
            self.alpha = 1
            self.annotators = annotators
        else:
            self.model = SegmentationModel()

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
        vis_train_images = config['data']['visualize_images']['train']
        save_image_color_legend()

        # Optimizer
        if config['data']['crowd'] and config['model']['crowd_type']!='prob-unet':
            optimizer = torch.optim.Adam([
                {'params': model.seg_model.parameters()},
                {'params': model.crowd_layers.parameters(), 'lr': 1e-3}
            ], lr=learning_rate)
        elif config['model']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam([
                dict(params=model.parameters(), lr=learning_rate),
            ])
        elif config['model']['optimizer'] == 'sgd_mom':
            optimizer = torch.optim.SGD([
                dict(params=model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True),
            ])
        else:
            raise Exception('Choose valid optimizer!')

        min_trace = config['model']['min_trace']

        # Training loop
        for i in range(0, epochs):

            print('\nEpoch: {}'.format(i))
            model.train()

            # Stop of the warm-up period
            if i == 5: #10 for cr_image_dice // 5 rest of the methods
                print("Minimize trace activated!")
                min_trace = True
                self.alpha = config['model']['alpha']
                print("Alpha updated", self.alpha)

                if config['data']['crowd'] and config['model']['crowd_type']!='prob-unet':
                    optimizer = torch.optim.Adam([
                        {'params': model.seg_model.parameters()},
                        {'params': model.crowd_layers.parameters(), 'lr': 1e-4}
                    ], lr=learning_rate)

            # Training in batches
            for j, (images, labels, imagename, ann_ids) in enumerate(trainloader):
                # Loading data to GPU
                images = images.cuda().float()
                labels = labels.cuda().long()
                ann_ids = ann_ids.cuda().float()

                # zero the parameter gradients
                optimizer.zero_grad()

                if config['data']['ignore_last_class']:
                    ignore_index = int(config['data']['class_no'])  # deleted class is always set to the last index
                else:
                    ignore_index = -100  # this means no index ignored
                self.ignore_index = ignore_index

                # Foward+loss (crowd or not)
                if config['model']['crowd_type'] == 'prob-unet':
                    _, labels = torch.max(labels, dim=1)
                    labels = labels[:,None,:,:]
                    model.forward(images, labels, training=True)
                    elbo = model.elbo(labels)
                    reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(
                        model.fcomb.layers)
                    loss = -elbo + 1e-5 * reg_loss
                elif config['data']['crowd']:
                    _, labels = torch.max(labels, dim=1)
                    y_pred, cms = model(images, ann_ids)
                    loss, loss_ce, loss_trace = noisy_label_loss(y_pred, cms, labels, ignore_index,
                                                                 min_trace, self.alpha, self.loss_mode)
                else:
                    _, labels = torch.max(labels, dim=1)
                    y_pred = model(images)

                    if self.loss_mode == 'ce':
                        loss = torch.nn.NLLLoss(reduction='mean', ignore_index=ignore_index, weight=class_weights)(
                            torch.log(y_pred+eps), labels)
                    elif self.loss_mode == 'dice':
                        loss = DiceLoss(ignore_index=ignore_index, from_logits=False, mode='multiclass')(
                            y_pred, labels)
                    elif self.loss_mode == 'focal':
                        loss = FocalLoss(reduction='mean', ignore_index=ignore_index, mode='multiclass')(
                            y_pred, labels)

                # Final prediction
                if not config['data']['crowd']:
                    _, y_pred_max = torch.max(y_pred[:, 0:class_no], dim=1)

                # Backprop
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

                # Save results in training (only save for not crowd methods)
                if j % int(config['logging']['interval']) == 0:
                    print("Iter {}/{} - batch loss : {:.4f}".format(j, len(trainloader), loss))
                    if not config['data']['crowd']:
                        train_results = self.get_results(y_pred_max, labels)
                        log_results(train_results, mode='train', step=(i * len(trainloader) * batch_s + j))
                        for k in range(len(imagename)):
                            if imagename[k] in vis_train_images:
                                labels_save = labels[k].cpu().detach().numpy()
                                y_pred_max_save = y_pred_max[k].cpu().detach().numpy()
                                images_save = images[k]  # .cpu().detach().numpy()
                                save_test_images(images_save, y_pred_max_save, labels_save, imagename[k], 'train')

            # Save validation results
            val_results = self.evaluate(validateloader, mode='val')  # TODO: validate crowd
            log_results(val_results, mode='val', step=int((i + 1) * len(trainloader) * batch_s))
            mlflow.log_metric('finished_epochs', i + 1, int((i + 1) * len(trainloader) * batch_s))

            # Save model
            metric_for_saving = val_results['macro_dice']
            if max_score < metric_for_saving and i > 10:
                save_model(model)
                max_score = metric_for_saving

            # LR decay
            if i > config['model']['lr_decay_after_epoch']:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / (1 + config['model']['lr_decay_param'])

            # Show annotator matrix
            if config['data']['crowd'] and config['model']['crowd_type']!='prob-unet':
                _,  ann_id = torch.max(ann_ids, dim=1)
                for ann_ix, cm in enumerate(cms):
                    cm = cm.view(5,5,512,512)
                    cm_ = cm[:,:,100,100]
                    cm_ = cm_/cm_.sum(0)
                    print("Annotators", ann_id)
                    print("CM ", ann_id[ann_ix].cpu().detach().numpy()+1, ": ", cm_.cpu().detach().numpy())

        if globals.config['data']['sr_experiment']:
            self.evaluate_sr(testloader)

        # Final evaluation of crowd
        if config['data']['crowd'] and config['model']['crowd_type']!='prob-unet':
            self.evaluate_crowd(trainloader, mode='train')

    def test(self, testloader):
        save_image_color_legend()
        results = self.evaluate(testloader)
        log_results(results, mode='test', step=None)
        save_results(results)

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
            for j, (test_img, test_label, test_name, _) in enumerate(evaluatedata):
                test_img = test_img.to(device=device, dtype=torch.float32)
                if config['model']['crowd_type'] == 'prob-unet':
                    model.forward(test_img, None, training=False)
                    test_pred = model.sample(testing=True)
                elif globals.config['data']['crowd']:
                    test_pred, _ = model(test_img)
                else:
                    test_pred = model(test_img)
                _, test_pred = torch.max(test_pred[:, 0:class_no], dim=1)
                test_pred_np = test_pred.cpu().detach().numpy()
                test_label = test_label.cpu().detach().numpy()
                test_label = np.argmax(test_label, axis=1)

                preds.append(test_pred_np.astype(np.int8).copy().flatten())
                labels.append(test_label.astype(np.int8).copy().flatten())

                for k in range(len(test_name)):
                    if test_name[k] in vis_images or vis_images == 'all':
                        img = test_img[k]
                        save_test_images(img, test_pred_np[k], test_label[k], test_name[k], mode)

            preds = np.concatenate(preds, axis=0, dtype=np.int8).flatten()
            labels = np.concatenate(labels, axis=0, dtype=np.int8).flatten()

            results = self.get_results(preds, labels)

            print('RESULTS for ' + mode)
            print(results)
            return results

    def evaluate_crowd(self, evaluatedata, mode='train'):
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
                if config['model']['crowd_type'] == 'pixel':
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
                    if config['model']['crowd_type'] == 'pixel':
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

        metrics_names = ['macro_dice', 'micro_dice', 'miou', 'accuracy']
        for class_id in range(class_no):
            metrics_names.append('dice_class_' + str(class_id) + '_' + class_names[class_id])

        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy().copy().flatten()
        if torch.is_tensor(label):
            label = label.cpu().detach().numpy().copy().flatten()

        results = segmentation_scores(label, pred, metrics_names)

        return results
