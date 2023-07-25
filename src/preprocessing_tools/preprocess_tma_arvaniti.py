import argparse
import numpy as np
from preprocessing_tools.preprocessing_utils import resize_all_images, create_voting_masks, \
    create_crossvalidation_splits, convert_to_rgb, calculate_dataset_statistics, convert_dataset_structure

import SimpleITK as sitk

CLASS_COLORS_BGR = [[128, 255, 96], [32, 224, 255], [0, 104, 255], [0, 0, 255], [255, 255, 255]]


parser = argparse.ArgumentParser(description="Preprocess prostate TMA dataset TMA Arvaniti")
parser.add_argument("--input_dir", "-i", type=str,
                    default="/data/BasesDeDatos/Arvaniti_TMA/original_dataset/",
                    help="Input directory of dataset.")
parser.add_argument("--output_dir", "-o", type=str,
                    default="/data/BasesDeDatos/Arvaniti_TMA/resized_dataset_512/",
                    help="Output directory of resized images.")
args = parser.parse_args()

config = vars(args)

dataset_specific_names = {'restructured_dir': "/data/BasesDeDatos/Arvaniti_TMA/restructured_dataset/",
                          'arvaniti_train_masks_dir' : 'Gleason_masks_train/',
                          'arvaniti_test_masks_dir1': 'Gleason_masks_test_pathologist1/',
                          'arvaniti_test_masks_dir2': 'Gleason_masks_test_pathologist2/',
                          'arvaniti_img_dirs': ['ZT76_39_A/', 'ZT76_39_B/', 'ZT80_38_A/', 'ZT80_38_B/', 'ZT80_38_C/',
                                                'ZT111_4_A/', 'ZT111_4_B/', 'ZT111_4_C/', 'ZT199_1_A/',
                                                'ZT199_1_B/', 'ZT204_6_A/', 'ZT204_6_B/']}
config.update(dataset_specific_names)


dataset_restructured_names = {'train_img_dir': 'Train_imgs/',
                              'test_img_dir': 'Test_imgs/',
                              'map_dir': 'Maps/',
                              'map_annotator_dirs': ['Maps1_T/', 'Maps2_T/']}
config.update(dataset_restructured_names)

config.update({'resize_resolution': 512})

def convert_masks(mask):
    # The initial classes in BGR are
    # white [255, 255, 255] (background), green [0, 255, 0] (normal tissue), blue [255, 0, 0] (GG3),
    # yellow [0, 255, 255] (GG4), red [0, 0, 255] (GG5)
    # We move these classes to: 0 (normal tissue), 1 (GG3), 2 (GG4), 3 (GG5), 4 (background)
    w, h = mask.shape[0], mask.shape[1]
    mask_new = np.ones(shape=[w, h, 1], dtype=np.uint) * -1

    def set_class_value(in_mask, original_rgb_value, target_mask, new_mask_value):
        ids = (in_mask == original_rgb_value).all(axis=2)
        target_mask[ids] = int(new_mask_value)
        return target_mask

    # Opencv uses BGR format
    mask_new = set_class_value(mask, [255, 255, 255], mask_new, 4) # Background
    mask_new = set_class_value(mask, [0, 255, 0], mask_new, 0) # Normal tissue
    mask_new = set_class_value(mask, [255, 0, 0], mask_new, 1) # GG3
    mask_new = set_class_value(mask, [0, 255, 255], mask_new, 2) # GG4
    mask_new = set_class_value(mask, [0, 0, 255], mask_new, 3) # GG5
    if np.any(mask_new < 0):
        print('Unknown value: ' )
        print(mask[mask_new < 0])
        raise Exception('Unknown RGB value')

    mask_new = np.squeeze(mask_new).astype(np.uint)
    return mask_new

mask_fct = convert_masks

gg5_list = ['ZT80_38_B_1_2.png', 'ZT80_38_B_7_4.png', 'ZT80_38_A_7_1.png', 'ZT80_38_A_3_7.png', 'ZT80_38_C_7_10.png',
            'ZT80_38_A_6_5.png', 'ZT80_38_A_8_3.png', 'ZT80_38_A_4_3.png', 'ZT80_38_B_7_7.png', 'ZT80_38_A_8_2.png',
            'ZT80_38_C_1_10.png', 'ZT80_38_B_3_1.png', 'ZT80_38_B_2_1.png', 'ZT80_38_B_2_2.png', 'ZT80_38_B_2_12.png',
            'ZT80_38_A_6_7.png', 'ZT80_38_C_5_8.png', 'ZT80_38_A_1_11.png', 'ZT80_38_C_2_1.png', 'ZT80_38_B_1_9.png',
            'ZT80_38_C_4_1.png', 'ZT80_38_A_1_8.png', 'ZT80_38_A_1_7.png']


# convert_dataset_structure(config)

# resize_all_images(config, config['restructured_dir'], mask_fct)
#
create_crossvalidation_splits(config, config['output_dir'] + config['test_img_dir'], gg5_list, 'Maps1_T/')

calculate_dataset_statistics(args.output_dir + args.map_dir + 'Maps2_T/', 'total')








