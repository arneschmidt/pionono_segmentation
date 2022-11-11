import argparse
import os
import cv2
import shutil
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser(description="Resize Images of prostate TMA")
parser.add_argument("--input_dir", "-i", type=str,
                    default="/data/BasesDeDatos/Gleason_2019/original_dataset/",
                    help="Input directory of dataset.")
parser.add_argument("--output_dir", "-o", type=str,
                    default="/data/BasesDeDatos/Gleason_2019/resized_dataset_1024/",
                    help="Output directory of resized images.")
args = parser.parse_args()

train_img_dir = 'Train_imgs/'
test_img_dir = 'Test_imgs/'
map_dir = 'Maps/'
map_annotator_dirs = ['Maps1_T/', 'Maps2_T/', 'Maps3_T/', 'Maps4_T/', 'Maps5_T/', 'Maps6_T/']


def resize_images_in_folder(in_dir, out_dir, resize_type='nearest', mask=False):
    print('### Resize ###')
    resize_res = 1024
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
    for img_file in img_file_list:
        img_path_in = in_dir + img_file
        img_path_out = out_dir + img_file
        img_path_out = img_path_out.replace('.jpg', '.png')
        img_path_out = img_path_out.replace('_classimg_nonconvex.png', '.png')

        image = cv2.imread(img_path_in)
        print('Image: ' + img_file + ' Shape: ' + str(image.shape))
        resolution_list.append(image.shape)
        image = cv2.resize(image, (resize_res, resize_res), interpolation=interpolation)

        if mask:
            # The initial classes are 0 (background), 1 (normal tissue), 3 (GG3), 4 (GG4), 5 (GG5), 6 (normal tissue)
            # We move these classes to: 0 (normal tissue), 1 (GG3), 2 (GG4), 3 (GG5), 4 (background)
            ones = np.ones_like(image)

            image = image - 2

            image = np.where(image==255, ones*0, image)
            image = np.where(image == 4, ones*0, image)
            image = np.where(image==254, ones*4, image)
            assert np.all(image >= 0)
            assert np.all(image <= 4)

        cv2.imwrite(img_path_out, image)
    # print('Image shapes: ' + np.unique(resolution_list))

def resize_all_images():
    # resize_images_in_folder(args.input_dir + train_img_dir, args.output_dir + train_img_dir, 'bicubic')
    # resize_images_in_folder(args.input_dir + test_img_dir, args.output_dir + test_img_dir, 'bicubic')
    for annotator_dir in map_annotator_dirs:
        resize_images_in_folder(args.input_dir + map_dir + annotator_dir, args.output_dir + map_dir + annotator_dir, 'nearest', mask=True)

def calculate_dataset_statistics():
    print('### Dataset Statistic ###')
    majority_dir_name = 'Majority_Voting/'
    majority_voting_path = args.output_dir + map_dir + majority_dir_name
    img_file_list = os.listdir(majority_voting_path)

    count_pixels = [0, 0, 0, 0, 0]
    count_classes = [0, 0, 0, 0, 0]
    for img_file in img_file_list:
        mask = cv2.imread(majority_voting_path + img_file)
        for c in range(len(count_pixels)):
            pixels_c = int(np.sum(mask==c)/3)
            count_pixels[c] += pixels_c
            if pixels_c > 0:
                count_classes[c] += 1
    n_all_pixels = np.sum(count_pixels)
    class_weights = n_all_pixels / (len(count_pixels) * np.array(count_pixels))
    print('Overall classes per pixel: ' + str(count_pixels))
    print('Overall classes per image: ' + str(count_classes))
    print('Class_weights: ' + str(class_weights))


def create_crossvalidation_splits():
    print('### Create Cross Validation ###')
    num_splits = 3
    percentage_validation = 0.3
    np.random.seed(0)
    train_img_path = args.output_dir + train_img_dir
    img_file_list = os.listdir(train_img_path)
    print('No of initial train images: ' + str(len(img_file_list)))
    for i in range(num_splits):
        print('Split ' + str(i))
        val_img_list = np.random.choice(img_file_list, size=int(percentage_validation*len(img_file_list)), replace=False)
        crossval_dir = args.output_dir + 'Crossval' + str(i) + '/'
        shutil.rmtree(crossval_dir, ignore_errors=True)
        os.makedirs(crossval_dir, exist_ok=True)
        crossval_dir_train = crossval_dir + 'train/'
        os.makedirs(crossval_dir_train, exist_ok=True)
        crossval_dir_val = crossval_dir + 'val/'
        os.makedirs(crossval_dir_val, exist_ok=True)
        for img in img_file_list:
            if img in val_img_list:
                shutil.copy(train_img_path + img, crossval_dir_val + img)
            else:
                shutil.copy(train_img_path + img, crossval_dir_train + img)
        print('No of val images: ' + str(len(os.listdir(crossval_dir_val))))
        print('No of train images: ' + str(len(os.listdir(crossval_dir_train))))

def create_majority_voting_masks():
    print('### Create Majority Voting ###')
    majority_dir_name = 'Majority_Voting/'
    majority_voting_path = args.output_dir + map_dir + majority_dir_name
    os.makedirs(majority_voting_path, exist_ok=True)
    train_img_path = args.output_dir + train_img_dir
    test_img_path = args.output_dir + test_img_dir
    all_imgs = np.concatenate([os.listdir(train_img_path), os.listdir(test_img_path)], axis=0)
    counter = 0
    for img in all_imgs:
        print(img)
        masks = []
        for a in range(len(map_annotator_dirs)):
            mask_path_in = args.output_dir + map_dir + map_annotator_dirs[a] + img
            image_array = cv2.imread(mask_path_in)
            if image_array is not None:
                masks.append(image_array)
        masks = np.array(masks)
        if masks.shape[0] > 0:
            mod = stats.mode(masks, axis=0)

            img_path_out = majority_voting_path + img
            cv2.imwrite(img_path_out, mod[0][0])
            counter = counter + 1
    print('Total images with availabel majority voting: ' + str(counter))


# resize_all_images()
#
# create_crossvalidation_splits()
#
# create_majority_voting_masks()

calculate_dataset_statistics()







