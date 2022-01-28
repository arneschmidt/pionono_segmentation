import torch

import src.utils.globals as globals
from src.utils.saving import save_model, save_results, save_test_image
from src.utils.model_architecture import create_model
from src.utils.test_helpers import segmentation_scores


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
        c_weights = [44, 0.4756, 0.5844, 1.5684, 3.1598, 4.7606]
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
                outputs_logits = model(images)
                y_pred = torch.softmax(outputs_logits, dim=1)
                _, labels = torch.max(labels, dim=1)
                loss = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0, weight=class_weights)(y_pred, labels)
                loss.backward()
                optimizer.step()

                if j % 10 == 0:
                    print("Iter {}/{} - batch loss : {:.4f}".format(j, len(trainloader), loss))

            train_dice, train_macro_dice, train_micro_dice = self.evaluate(trainloader)
            dice_val_class, validate_macro_dice, validate_micro_dice = self.evaluate(validateloader)
            print(
                '**************** Epoch {}/{},'
                'Train macro dice: {:.4f}, '
                'Train micro dice: {:.4f}, '
                'Val macro dice: {:.4f},'
                'Val micro dice: {:.4f},'.format(i + 1, epochs,
                                                 train_macro_dice,
                                                 train_micro_dice,
                                                 validate_macro_dice,
                                                 validate_micro_dice))
            print('Val dice per class:', *dice_val_class, sep='\n- ')
            print('Train dice per class:', *train_dice, sep='\n-')

            if max_score < validate_macro_dice:
                save_model(model)

            if i > config['model']['lr_decay_after_epoch']:
                for g in optimizer.param_groups:
                    g['lr'] = 1 / (g['lr'] + config['model']['lr_decay_param'])

    def evaluate(self, evaluatedata):
        config = globals.config
        class_no = config['data']['class_no']
        model = self.model
        device = self.device
        model.eval()

        with torch.no_grad():

            test_dice = 0
            test_macro_dice = 0
            test_micro_dice = 0

            for j, (testimg, testlabel, testname) in enumerate(evaluatedata):

                testimg = testimg.to(device=device, dtype=torch.float32)
                testlabel = testlabel.to(device=device, dtype=torch.float32)

                testoutput = model(testimg)
                _, testoutput = torch.max(testoutput[:, 1:], dim=1)
                _, testlabel = torch.max(testlabel, dim=1)

                mean_dice, mean_macro_dice, mean_micro_dice = segmentation_scores(testlabel.cpu().detach().numpy(),
                                                                                  testoutput.cpu().detach().numpy(),
                                                                                  class_no - 1)
                test_dice += mean_dice
                test_macro_dice += mean_macro_dice
                test_micro_dice += mean_micro_dice
            #
            return test_dice / (j + 1), test_macro_dice / (j + 1), test_micro_dice / (j + 1)

    def test(self, testdata):
        config = globals.config
        class_no = config['data']['class_no']
        model = self.model
        device = self.device
        model.eval()

        with torch.no_grad():

            test_dice_sum = 0
            test_macro_dice_sum = 0
            test_micro_dice_sum = 0

            for j, (test_images, test_label, test_name) in enumerate(testdata):

                test_images = test_images.to(device=device, dtype=torch.float32)
                test_label = test_label.to(device=device, dtype=torch.float32)
                _, test_label = torch.max(test_label, dim=1)

                test_preds = model(test_images)
                _, test_preds = torch.max(test_preds[:, 1:], dim=1)

                mean_dice, mean_macro_dice, mean_micro_dice = segmentation_scores(test_label.cpu().detach().numpy(),
                                                                                  test_preds.cpu().detach().numpy(),
                                                                                  class_no - 1)
                test_dice_sum += mean_dice
                test_macro_dice_sum += mean_macro_dice
                test_micro_dice_sum += mean_micro_dice

                test_label = test_label.cpu().detach().numpy()
                test_preds = test_preds.cpu().detach().numpy()
                for image_name in test_name:
                    if image_name in config['data']['visualize_test_images'] or config['data']['visualize_test_images'] == 'all':
                        save_test_image(test_preds, test_label, test_name)

            n = len(testdata)
            results = {
                'macro_dice': test_macro_dice_sum / n,
                'micro_dice': test_micro_dice_sum / n,
                'dice_per_class': test_dice_sum / n
            }
            save_results(results)