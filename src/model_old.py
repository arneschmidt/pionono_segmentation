import os
import torch
import imageio
import errno
import numpy as np


def segmentation_scores(label_trues, label_preds, n_class):
    '''
    :param label_trues:
    :param label_preds:
    :param n_class:
    :return:
    '''
    assert len(label_trues) == len(label_preds)

    label_preds = label_preds[label_trues!=0]
    label_trues = label_trues[label_trues!=0]

    #label_trues += 1
    #label_preds += 1
    label_preds +=1 # We are not considering class '0'
    label_preds = np.asarray(label_preds, dtype='int8').copy()
    label_trues = np.asarray(label_trues, dtype='int8').copy()

    label_preds = label_preds * (label_trues > 0)

    intersection = label_preds * (label_preds == label_trues)
    (area_intersection, _) = np.histogram(intersection, bins=n_class, range=(1, n_class))
    (area_pred, _) = np.histogram(label_preds, bins=n_class, range=(1, n_class))
    (area_lab, _) = np.histogram(label_trues, bins=n_class, range=(1, n_class))
    area_sum = area_pred + area_lab
    #

    dice = ((2 * area_intersection + 1e-6) / (area_sum + 1e-6))
    macro_dice = dice.mean()

    intersection = (label_preds == label_trues).sum(axis=None)
    sum_ = 2 * np.prod(label_preds.shape)
    micro_dice = ((2 * intersection + 1e-6) / (sum_ + 1e-6))

    return dice, macro_dice, micro_dice


def evaluate(evaluatedata, model, device, class_no):
    """
    Args:
        evaluatedata:
        model:
        device:
        class_no:
    Returns:
    """
    model.eval()
    #
    with torch.no_grad():
        #
        test_dice = 0
        test_macro_dice = 0
        test_micro_dice = 0
        #
        for j, (testimg, testlabel, testname) in enumerate(evaluatedata):
            #
            testimg = testimg.to(device=device, dtype=torch.float32)
            testlabel = testlabel.to(device=device, dtype=torch.float32)
            #
            testoutput = model(testimg)
            _, testoutput = torch.max(testoutput[:, 1:], dim=1)
            _, testlabel = torch.max(testlabel, dim=1)
            #
            mean_dice, mean_macro_dice, mean_micro_dice = segmentation_scores(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no-1)
            test_dice += mean_dice
            test_macro_dice += mean_macro_dice
            test_micro_dice += mean_micro_dice
        #
        return test_dice / (j+1), test_macro_dice / (j+1), test_micro_dice / (j+1)


def test(testdata,
         model,
         device,
         class_no,
         save_path):
    """
    Args:
        testdata:
        model:
        device:
        class_no:
        save_path:
    Returns:
    """
    model.eval()

    with torch.no_grad():
        #
        test_dice = 0
        test_macro_dice = 0
        test_micro_dice = 0
        #
        for j, (testimg, testlabel, testname) in enumerate(testdata):
            #
            testimg = testimg.to(device=device, dtype=torch.float32)
            testlabel = testlabel.to(device=device, dtype=torch.float32)
            _, testlabel = torch.max(testlabel, dim=1)
            #
            testoutput = model(testimg)
            _, testoutput = torch.max(testoutput[:, 1:], dim=1)
            #
            mean_dice, mean_macro_dice, mean_micro_dice = segmentation_scores(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no-1)
            test_dice += mean_dice
            test_macro_dice += mean_macro_dice
            test_micro_dice += mean_micro_dice
            #
            # ========================================================
            # # Plotting segmentation:
            # ========================================================
            prediction_map_path = save_path + '/' + 'Visual_results'
            #
            try:
                os.mkdir(prediction_map_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            testlabel = testlabel.cpu().detach().numpy()

            b, h, w = np.shape(testlabel)

            testoutput_original = np.asarray(testoutput.cpu().detach().numpy(), dtype=np.uint8) + 1
            #

            segmentation_map = 2 * np.zeros((h, w, 3), dtype=np.uint8)


            for i, test_out in enumerate(testoutput_original):
                    # multi class for breast cancer
                    segmentation_map[:, :, 0][test_out == 1] = 153
                    segmentation_map[:, :, 1][test_out == 1] = 0
                    segmentation_map[:, :, 2][test_out == 1] = 0

                    segmentation_map[:, :, 0][test_out == 2] = 255
                    segmentation_map[:, :, 1][test_out == 2] = 102
                    segmentation_map[:, :, 2][test_out == 2] = 204

                    segmentation_map[:, :, 0][test_out == 3] = 0
                    segmentation_map[:, :, 1][test_out == 3] = 153
                    segmentation_map[:, :, 2][test_out == 3] = 51

                    segmentation_map[:, :, 0][test_out == 4] = 153
                    segmentation_map[:, :, 1][test_out == 4] = 0
                    segmentation_map[:, :, 2][test_out == 4] = 204

                    segmentation_map[:, :, 0][test_out == 5] = 0
                    segmentation_map[:, :, 1][test_out == 5] = 179
                    segmentation_map[:, :, 2][test_out == 5] = 255


                    segmentation_map[:, :, 0][testlabel[i] == 0] = 0
                    segmentation_map[:, :, 1][testlabel[i] == 0] = 0
                    segmentation_map[:, :, 2][testlabel[i] == 0] = 0


                    prediction_name = 'pred_seg_' + testname[i]
                    full_error_map_name = os.path.join(prediction_map_path, prediction_name)
                    imageio.imsave(full_error_map_name, segmentation_map)

        #
        prediction_result_path = save_path + '/Quantitative_Results'
        #
        try:
            #
            os.mkdir(prediction_result_path)
            #
        except OSError as exc:
            #
            if exc.errno != errno.EEXIST:
                #
                raise
            #
            pass
        #

        dice_class = test_dice / len(testdata)
        print('Dice per class:', *dice_class, sep='\n- ')
        print('Macro dice: ', test_macro_dice / len(testdata))
        print('Micro dice: ', test_micro_dice / len(testdata))
        result_dictionary = {'Test macro': str(test_macro_dice / len(testdata)),
                            'Test micro': str(test_micro_dice / len(testdata))}
        #
        ff_path = prediction_result_path + '/test_result_data.txt'
        ff = open(ff_path, 'w')
        ff.write(str(result_dictionary))
        ff.write(" ".join(str(x) for x in dice_class))
        ff.close()

