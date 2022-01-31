import torch
import mlflow
import numpy as np
import src.utils.globals as globals
from src.utils.saving import save_model, save_results, save_test_images
from src.utils.model_architecture import create_model
from src.utils.test_helpers import segmentation_scores
from src.utils.logging import log_results


class SegmentationModel():
    def __init__(self):
        self.model = create_model()
        if torch.cuda.is_available():
            print('Running on GPU')
            self.device = torch.device('cuda')
        else:
            print('Running on CPU')
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

        if config['model']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam([
                dict(params=model.parameters(), lr=learning_rate),
            ])
        elif config['model']['optimizer'] == 'sgd_mom':
            optimizer = torch.optim.SGD([
                dict(params=model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True),
            ])
        else:
            raise Exception('Choose valid optimizer!')


        for i in range(0, epochs):

            print('\nEpoch: {}'.format(i))
            model.train()

            for j, (images, labels, imagename) in enumerate(trainloader):
                images = images.cuda().float()  # to(device=device, dtype=torch.float32)
                labels = labels.cuda().long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                y_pred = model(images)
                # y_pred = torch.softmax(outputs_logits, dim=1)
                _, labels = torch.max(labels, dim=1)
                if config['data']['ignore_last_class']:
                    ignore_index = int(config['data']['class_no']) # deleted class is always set to the last index
                else:
                    ignore_index = -100 # this means no index ignored
                loss = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index, weight=class_weights)(y_pred, labels)
                loss.backward()
                optimizer.step()

                if j % 10 == 0:
                    print("Iter {}/{} - batch loss : {:.4f}".format(j, len(trainloader), loss))

            train_results = self.evaluate(trainloader, mode = 'train')
            val_results = self.evaluate(validateloader, mode = 'val')

            log_results(train_results, mode = 'train', step=i)
            log_results(val_results, mode = 'val', step=i)

            metric_for_saving = val_results['macro_dice']
            if max_score < metric_for_saving:
                save_model(model)
                max_score = metric_for_saving

            if i > config['model']['lr_decay_after_epoch']:
                for g in optimizer.param_groups:
                    g['lr'] = 1 / (g['lr'] + config['model']['lr_decay_param'])

    def test(self, testloader):
        results = self.evaluate(testloader)
        log_results(results, mode = 'val', step=None)
        save_results(results)

    def evaluate(self, evaluatedata, mode='test'):
        config = globals.config
        class_no = config['data']['class_no']
        model = self.model
        device = self.device
        model.eval()

        with torch.no_grad():
            test_dice_sum = np.zeros(shape=class_no)
            test_macro_dice_sum = 0
            test_micro_dice_sum = 0

            for j, (test_img, test_label, test_name) in enumerate(evaluatedata):

                test_img = test_img.to(device=device, dtype=torch.float32)
                test_label = test_label.to(device=device, dtype=torch.float32)
                _, test_label = torch.max(test_label[:, 0:class_no], dim=1)

                test_pred = model(test_img)
                _, test_pred = torch.max(test_pred[:, 0:class_no], dim=1)


                test_label = test_label.cpu().detach().numpy()
                test_pred = test_pred.cpu().detach().numpy()

                mean_dice, mean_macro_dice, mean_micro_dice = segmentation_scores(test_label,
                                                                                  test_pred,
                                                                                  class_no)
                test_dice_sum += mean_dice
                test_macro_dice_sum += mean_macro_dice
                test_micro_dice_sum += mean_micro_dice

                if mode != 'train':
                    for image_name in test_name:
                        if image_name in config['data']['visualize_images'][mode] or config['data']['visualize_images'][mode] == 'all':
                            save_test_images(test_pred, test_label, test_name, mode)

            n = len(evaluatedata)
            results = {
                'macro_dice': test_macro_dice_sum / n,
                'micro_dice': test_micro_dice_sum / n,
            }

            for i in range(test_dice_sum.size):
                results['dice_class_' + str(i)] = test_dice_sum[i]/n

            print('RESULTS for ' + mode)
            print(results)
            return results
