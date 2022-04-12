import torch
import mlflow
import warnings
import numpy as np
import utils.globals as globals
from utils.saving import save_model, save_results, save_test_images, save_image_color_legend, save_crowd_images
from utils.model_architecture import SegmentationModel
from utils.crowd_model_architecture import Crowd_segmentationModel
from utils.loss import noisy_label_loss
from utils.test_helpers import segmentation_scores
from utils.logging import log_results


class ModelHandler():
    def __init__(self, annotators):
        config = globals.config
        if config['data']['crowd']:
            self.model = Crowd_segmentationModel(annotators)
            self.alpha = config['model']['alpha']
            self.annotators = annotators
        else:
            self.model = SegmentationModel()
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
        # print(vis_train_images)
        save_image_color_legend()

        """
        if config['model']['optimizer'] == 'adam':
            if config['data']['crowd']:
                print("Crowd optimization")
                optimizer = torch.optim.Adam([
                    {'params': model.seg_model.parameters()},
                    {'params': model.spatial_cms.parameters(), 'lr': 1e-2}
                ], lr=learning_rate)
        """

        if config['data']['crowd']:
            optimizer= torch.optim.Adam([
                {'params': model.seg_model.parameters()},
                {'params': model.crowd_layers.parameters(), 'lr': 1e-2}
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

        for i in range(0, epochs):

            print('\nEpoch: {}'.format(i))
            model.train()

            if i > 50:
                min_trace = True

            for j, (images, labels, imagename, ann_ids) in enumerate(trainloader):
                # print(images.shape, labels.shape)
                # print(imagename)

                # images =  images.permute(0,3,1,2)
                images = images.cuda().float()  # to(device=device, dtype=torch.float32)
                #print("labels", labels.shape)
                labels = labels.cuda().long()
                ann_ids = ann_ids.cuda().float()
                #print("Ann_IDs: ", ann_ids)

                # zero the parameter gradients
                optimizer.zero_grad()

                if config['data']['ignore_last_class']:
                    ignore_index = int(config['data']['class_no'])  # deleted class is always set to the last index
                else:
                    ignore_index = -100  # this means no index ignored

                self.ignore_index = ignore_index
                if config['data']['crowd']:
                    # _, labels = torch.max(labels, dim=2)

                    _, labels = torch.max(labels, dim=1)
                    # labels = labels.permute(1, 0, 2, 3)
                    # labels = torch.unsqueeze(labels, dim=2)
                    y_pred, cms = model(images, ann_ids)
                    loss, loss_ce, loss_trace = noisy_label_loss(y_pred, cms, labels, ignore_index,
                                                                 min_trace, self.alpha)
                else:
                    # forward + backward + optimize
                    _, labels = torch.max(labels, dim=1)
                    y_pred = model(images)
                    loss = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index, weight=class_weights)(
                        y_pred, labels)

                _, y_pred_max = torch.max(y_pred[:, 0:class_no], dim=1)

                loss.backward()
                optimizer.step()

                # TODO: save train in crowdsourcing
                if j % int(config['logging']['interval']) == 0:
                    if not config['data']['crowd']:
                        train_results = self.get_results(y_pred_max, labels)
                        log_results(train_results, mode='train', step=(i * len(trainloader) * batch_s + j))
                    print("Iter {}/{} - batch loss : {:.4f}".format(j, len(trainloader), loss))
                    if config['data']['crowd']:
                        continue
                        # self.evaluate_crowd(trainloader, mode='train', annotators=self.annotators)
                    else:
                        for k in range(len(imagename)):
                            if imagename[k] in vis_train_images:
                                labels_save = labels[k].cpu().detach().numpy()
                                y_pred_max_save = y_pred_max[k].cpu().detach().numpy()
                                images_save = images[k]  # .cpu().detach().numpy()
                                save_test_images(images_save, y_pred_max_save, labels_save, imagename[k], 'train')

            val_results = self.evaluate(validateloader, mode='val') #TODO: validate crowd
            log_results(val_results, mode='val', step=int((i + 1) * len(trainloader) * batch_s))
            mlflow.log_metric('finished_epochs', i + 1, int((i + 1) * len(trainloader) * batch_s))

            metric_for_saving = val_results['macro_dice']
            if max_score < metric_for_saving:
                save_model(model)
                max_score = metric_for_saving

            if i > config['model']['lr_decay_after_epoch']:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / (1 + config['model']['lr_decay_param'])

            if config['data']['crowd'] and config['model']['crowd_global']:
                for ann_ix, cm in enumerate(cms):
                    cm_ = cm[:, :, 0, 0]
                    #_, ann_id = torch.max(ann_ids[ann_ix])
                    print("CM NP", ann_ids[ann_ix], ": ", cm_)
                    # save_confusion_matrix(ann_ix, cm_)

    def test(self, testloader):
        save_image_color_legend()
        results = self.evaluate(testloader)
        log_results(results, mode='test', step=None)
        save_results(results)

    def evaluate(self, evaluatedata, mode='test'):
        config = globals.config
        class_no = config['data']['class_no']
        vis_images = config['data']['visualize_images'][mode]
        # print("vis_images", vis_images)
        model = self.model
        device = self.device
        model.eval()

        labels = []
        preds = []

        with torch.no_grad():
            for j, (test_img, test_label, test_name) in enumerate(evaluatedata):
                test_img = test_img.to(device=device, dtype=torch.float32)
                if globals.config['data']['crowd']:
                    test_pred, _ = model(test_img)
                else:
                    test_pred = model(test_img)
                _, test_pred = torch.max(test_pred[:, 0:class_no], dim=1)
                test_pred_np = test_pred.cpu().detach().numpy()
                test_label = test_label.cpu().detach().numpy()
                test_label = np.argmax(test_label, axis=1)

                preds.append(test_pred_np.astype(np.int8).copy().flatten())
                labels.append(test_label.astype(np.int8).copy().flatten())
                # print(test_name)
                for k in range(len(test_name)):
                    if test_name[k] in vis_images or vis_images == 'all':
                        img = test_img[k]  # .cpu().detach().numpy()
                        # print(test_name[k])
                        save_test_images(img, test_pred_np[k], test_label[k], test_name[k], mode)

            preds = np.concatenate(preds, axis=0, dtype=np.int8).flatten()
            labels = np.concatenate(labels, axis=0, dtype=np.int8).flatten()

            results = self.get_results(preds, labels)

            print('RESULTS for ' + mode)
            print(results)
            return results

    def evaluate_crowd(self, evaluatedata, mode='train', annotators=None):
        config = globals.config
        class_no = config['data']['class_no']
        vis_images = config['data']['visualize_images'][mode]
        # print("vis_images", vis_images)
        model = self.model
        device = self.device
        model.eval()

        with torch.no_grad():
            for j, (test_img, test_label, test_name, ann_id) in enumerate(evaluatedata):
                test_img = test_img.to(device=device, dtype=torch.float32)
                pred_noisy_list = []
                test_pred, cms = model(test_img)

                test_pred_np = test_pred.cpu().detach().numpy()
                test_pred_np = np.argmax(test_pred_np, axis=1)

                _, test_label = torch.max(test_label, dim=1)
                #test_label = test_label.permute(1, 0, 2, 3)
                test_label = test_label.cpu().detach().numpy()
                # print(test_pred.size())
                b, c, h, w = test_pred.size()
                for cm in cms:
                    # print(cm.shape)
                    pred_noisy = test_pred.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)

                    cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(
                        b * h * w, c, c)

                    # normalisation along the rows:
                    # print(cm.shape)
                    cm = cm / cm.sum(1, keepdim=True)
                    # print(cm[0])

                    # matrix multiplication to calculate the predicted noisy segmentation:
                    # cm: b*h*w x c x c
                    # pred_noisy: b*h*w x c x 1
                    pred_noisy = torch.bmm(cm, pred_noisy).view(b * h * w, c)
                    pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                    # print("shape noisy1", pred_noisy)
                    # print("shape noisy2", pred_noisy.size())
                    _, pred_noisy = torch.max(pred_noisy[:, 0:class_no], dim=1)
                    # print("shape noisy3", pred_noisy)
                    # print("shape noisy4", pred_noisy.size())
                    pred_noisy_list.append(pred_noisy.cpu().detach().numpy().astype(np.int8).copy().flatten())
                # all_preds_noisy.append(pred_noisy_list)  # B x L x 512 x 512

                # print(test_name)
                for k in range(len(test_name)):
                    if test_name[k] in vis_images or vis_images == 'all':
                        img = test_img[k]  # .cpu().detach().numpy()
                        # print(test_name[k])
                        for i, test_noisy_pred in enumerate(pred_noisy_list):
                            # print("parche", test_name[k])
                            if np.any(test_label[i][k] != self.ignore_index):
                                # print("Entra!")
                                # print(test_label.shape)
                                save_crowd_images(img, test_pred_np[k], test_noisy_pred[k], test_label[i][k],
                                             test_name[k], self.annotators[i])

            """" TODO: metrics for annotators
            preds = np.concatenate(preds, axis=0, dtype=np.int8).flatten()
            labels = np.concatenate(labels, axis=0, dtype=np.int8).flatten()

            results = self.get_results(preds, labels)

            print('RESULTS for ' + mode)
            print(results)
            return results
            """

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
