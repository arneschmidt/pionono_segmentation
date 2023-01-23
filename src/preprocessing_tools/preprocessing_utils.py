import os
import cv2
import shutil
import random
import numpy as np
from scipy import stats

import SimpleITK as sitk

CLASS_COLORS_BGR = [[128, 255, 96], [32, 224, 255], [0, 104, 255], [0, 0, 255], [255, 255, 255]]


def convert_dataset_structure(config):
    print('### Convert dataset structure ###')
    train_img_out_dir = config['restructured_dir'] + config['train_img_dir']
    test_img_out_dir = config['restructured_dir'] + config['test_img_dir']
    ann_1_masks_out_dir = config['restructured_dir'] + config['map_dir'] + config['map_annotator_dirs'][0]
    ann_2_masks_out_dir = config['restructured_dir'] + config['map_dir'] + config['map_annotator_dirs'][1]
    os.makedirs(train_img_out_dir, exist_ok=True)
    os.makedirs(test_img_out_dir, exist_ok=True)
    os.makedirs(ann_1_masks_out_dir, exist_ok=True)
    os.makedirs(ann_2_masks_out_dir, exist_ok=True)

    # copy masks, create list of images
    print('Copy masks')
    def copy_masks_and_create_list(mask_dir, out_dir):
        masks_list = os.listdir(mask_dir)
        img_list = []
        for m in  range(len(masks_list)):

            mask_name = masks_list[m]
            if '._' in mask_name:
                print('Drop erroneous mask' + mask_name)
                continue
            new_name = mask_name.replace('mask_', '')
            new_name = new_name.replace('mask1_', '')
            new_name = new_name.replace('mask2_', '')
            shutil.copy(mask_dir + mask_name, out_dir + new_name)

            img_name = new_name.replace('.png', '.jpg')
            img_list.append(img_name)

        return img_list

    train_img_list = copy_masks_and_create_list(config['input_dir'] + config['arvaniti_train_masks_dir'], ann_1_masks_out_dir)
    test_img_list1 = copy_masks_and_create_list(config['input_dir'] + config['arvaniti_test_masks_dir1'], ann_1_masks_out_dir)
    test_img_list2 = copy_masks_and_create_list(config['input_dir'] + config['arvaniti_test_masks_dir2'], ann_2_masks_out_dir)

    def check_duplicates(list, name):
        seen = set()
        dupes = [x for x in list if x in seen or seen.add(x)]
        if len(dupes) > 0:
            print('Duplicates in ' + name + ' :')
            print(dupes)
        return np.unique(list)

    train_img_list = check_duplicates(train_img_list, 'train_img_list')
    test_img_list1 = check_duplicates(test_img_list1, 'test_img_list1')
    test_img_list2 = check_duplicates(test_img_list2, 'test_img_list2')

    print('Found Train images A1: ' + str(len(train_img_list)) + ' Test images A1: ' + str(
        len(test_img_list1)) + ' Test images A2: ' + str(len(test_img_list2)))

    print('Copy images')
    # copy all images into new structure
    def copy_list_of_imgs(list, out_dir):
        for i in range(len(list)):
            img_name = list[i]
            copied = False
            for j in range(len(config['arvaniti_img_dirs'])):
                arv_img_dir = config['input_dir'] + config['arvaniti_img_dirs'][j]

                if img_name in os.listdir(arv_img_dir):
                    shutil.copy(arv_img_dir + img_name, out_dir + img_name)
                    copied = True
                    break
            if not copied:
                print('Image ' + img_name + ' not found!')

    # copy train images
    copy_list_of_imgs(train_img_list, train_img_out_dir)
    copy_list_of_imgs(test_img_list1, test_img_out_dir)
    copy_list_of_imgs(test_img_list2, test_img_out_dir)


def resize_images_in_folder(config, in_dir, out_dir, resize_type='nearest', mask_fct=None):
    print('### Resize ###')
    resize_res = config['resize_resolution']
    print('Processing input: ' + in_dir + ' Output: ' + out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if resize_type == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif resize_type == 'linear':
        interpolation = cv2.INTER_LINEAR
    elif resize_type == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        print('Choose valid interpolation!')

    img_file_list = os.listdir(in_dir)

    print('Images found:' + str(len(img_file_list)))
    resolution_list = []

    for i in range(len(img_file_list)):
        if i % 10 == 0:
            print('Image ' + str(i) + '/' + str(len(img_file_list)))
        img_file = img_file_list[i]
        img_path_in = in_dir + img_file
        img_path_out = out_dir + img_file
        img_path_out = img_path_out.replace('.jpg', '.png')
        img_path_out = img_path_out.replace('_classimg_nonconvex.png', '.png')

        image = cv2.imread(img_path_in)
        # print('Image: ' + img_file + ' Shape: ' + str(image.shape))
        resolution_list.append(image.shape)
        image = cv2.resize(image, (resize_res, resize_res), interpolation=interpolation)
        valid = True
        if mask_fct is not None:
            image = mask_fct(image)
            assert np.all(image >= 0)
            assert np.all(image <= 4)
            if np.all(image == 4):
                valid = False
        if valid:
            cv2.imwrite(img_path_out, image)
    # print('Image shapes: ' + np.unique(resolution_list))

def resize_all_images(config, input_dir, mask_fct):
    # resize_images_in_folder(config, input_dir + config['train_img_dir'], config['output_dir'] + config['train_img_dir'], 'bicubic')
    # resize_images_in_folder(config, input_dir + config['test_img_dir'], config['output_dir'] + config['test_img_dir'], 'bicubic')
    for annotator_dir in config['map_annotator_dirs']:
        resize_images_in_folder(config, input_dir + config['map_dir'] + annotator_dir,
                                config['output_dir'] + config['map_dir'] + annotator_dir, 'nearest', mask_fct=mask_fct)

def calculate_dataset_statistics(dir_path, name, list=None, file=None):
    print('--------- Dataset Statistic of ' + name, file=file)
    if list is None:
        img_file_list = os.listdir(dir_path)
    else:
        img_file_list = list

    count_pixels = [0, 0, 0, 0, 0]
    count_classes = [0, 0, 0, 0, 0]
    print('GG5 image list: ', file=file)
    for img_file in img_file_list:
        mask = cv2.imread(dir_path + img_file)
        for c in range(len(count_pixels)):
            pixels_c = int(np.sum(mask==c)/3)
            count_pixels[c] += pixels_c
            if pixels_c > 0:
                count_classes[c] += 1
                if c == 3:
                    print("- '" + img_file + "'", file=file)

    n_all_pixels = np.sum(count_pixels)
    class_weights = n_all_pixels / (len(count_pixels) * np.array(count_pixels))
    print('Overall classes per pixel: ' + str(count_pixels), file=file)
    print('Overall classes per image: ' + str(count_classes), file=file)
    print('Class_weights: ' + str(class_weights), file=file)
    print('---------')

def create_crossvalidation_splits(config, img_dir, list_gg5, data_stat_dir = 'STAPLE/'):
    print('### Create Cross Validation ###')
    num_splits = 4

    np.random.seed(0)
    img_file_list = os.listdir(img_dir)

    val_n = len(img_file_list)/num_splits

    # Egually distrbute GG5 images in the image list.
    # We assure that more or less the equal amount of GG5 images is in each crossvalidation split
    frequency_gg5 = len(img_file_list)/len(list_gg5)

    for img in list_gg5:
        img_file_list.remove(img)
    np.random.shuffle(img_file_list)


    for i in range(len(list_gg5)):
        index = int(frequency_gg5*i)
        img_file_list = np.insert(img_file_list, index, list_gg5[i])


    with open(config['output_dir'] + '/crossval_statistics.txt', 'w') as f:
        # Cut list into splits and save images
        print('No of initial images: ' + str(len(img_file_list)))
        for i in range(num_splits):
            print('Split ' + str(i), file=f)
            print('Split ' + str(i))
            val_start_id = int(val_n * i)
            val_stop_id = int(val_n * (i + 1))
            val_img_list = img_file_list[val_start_id:val_stop_id]
            crossval_dir = config['output_dir'] + 'Crossval' + str(i) + '/'
            shutil.rmtree(crossval_dir, ignore_errors=True)
            os.makedirs(crossval_dir, exist_ok=True)
            crossval_dir_train = crossval_dir + 'train/'
            os.makedirs(crossval_dir_train, exist_ok=True)
            crossval_dir_val = crossval_dir + 'val/'
            os.makedirs(crossval_dir_val, exist_ok=True)
            train_img_list = []
            for img in img_file_list:
                if img in val_img_list:
                    shutil.copy(img_dir + img, crossval_dir_val + img)
                else:
                    shutil.copy(img_dir + img, crossval_dir_train + img)
                    train_img_list.append(img)
            print('No of val images: ' + str(len(os.listdir(crossval_dir_val))), file=f)
            calculate_dataset_statistics(config['output_dir'] + config['map_dir'] + data_stat_dir, name='Validation', list=val_img_list, file=f)
            print('No of train images: ' + str(len(os.listdir(crossval_dir_train))), file=f)
            calculate_dataset_statistics(config['output_dir'] + config['map_dir'] + data_stat_dir, name='Training', list=train_img_list, file=f)


def create_voting_masks(config, voting_mechanism ='majority', dir_name='MV/'):
    """
    voting mecanism: 'majority' or 'staple'
    """
    print('### Create Voting Maps ###')
    print('Mode: ' + voting_mechanism)
    voting_path = config['output_dir'] + config['map_dir'] + dir_name
    os.makedirs(voting_path, exist_ok=True)
    train_img_path = config['output_dir'] + config['train_img_dir']
    test_img_path = config['output_dir'] + config['test_img_dir']
    all_imgs = np.concatenate([os.listdir(train_img_path), os.listdir(test_img_path)], axis=0)
    counter = 0
    for img in all_imgs:
        print(img)
        masks = []
        for a in range(len(config['map_annotator_dirs'])):
            mask_path_in = config['output_dir'] + config['map_dir'] + config['map_annotator_dirs'][a] + img
            image_array = cv2.imread(mask_path_in)
            if image_array is not None:
                masks.append(image_array)
        masks = np.array(masks)
        if masks.shape[0] > 0:
            if voting_mechanism == 'majority':
                vote_masks = stats.mode(masks, axis=0)[0][0]
            elif voting_mechanism == 'staple':
                masks_sitk_format = [sitk.GetImageFromArray(mask.astype(np.uint8)) for mask in masks]
                vote_masks_sitk_format = sitk.MultiLabelSTAPLE(masks_sitk_format)
                vote_masks = sitk.GetArrayFromImage(vote_masks_sitk_format)
                if np.any(vote_masks < 0) or np.any(vote_masks > 4) or np.any(np.mod(vote_masks,1) != 0):
                    # fall back to majority voting to avoid unannotated regions
                    vote_masks = stats.mode(masks, axis=0)[0][0]
            else:
                print('Choose valid voting mechanism')

            img_path_out = voting_path + img
            cv2.imwrite(img_path_out, vote_masks)
            counter = counter + 1
    print('Total images with voting: ' + str(counter))
    calculate_dataset_statistics(config['output_dir'] + config['map_dir'] + dir_name, voting_mechanism)

def convert_to_rgb(config, map_annotator_dirs):
    print('### Convert Maps to RGB images ###')
    rgb_dir = 'rgb_images/'
    rgb_path = config['output_dir'] + rgb_dir
    os.makedirs(rgb_path, exist_ok=True)
    map_dir = 'Maps/'

    os.makedirs(rgb_path, exist_ok=True)
    for m in range(len(map_annotator_dirs)):
        map_annotator_dir = map_annotator_dirs[m]
        in_dir = config['output_dir'] + map_dir + map_annotator_dir
        out_dir = rgb_path + map_annotator_dir
        os.makedirs(out_dir, exist_ok=True)
        img_file_list = os.listdir(in_dir)
        print(map_annotator_dir)
        print('Images found:' + str(len(img_file_list)))
        for img_file in img_file_list:
            img_path_in = in_dir + img_file
            img_path_out = out_dir + img_file

            image = cv2.imread(img_path_in)
            print('Image: ' + img_file + ' Shape: ' + str(image.shape))

            # classes : 0 (normal tissue), 1 (GG3), 2 (GG4), 3 (GG5), 4 (background)
            ones = np.ones_like(image)

            for c in range(len(CLASS_COLORS_BGR)):
                image = np.where(image == c, ones * CLASS_COLORS_BGR[c], image)

            assert np.all(image >= 0)
            assert np.all(image <= 255)

            cv2.imwrite(img_path_out, image)