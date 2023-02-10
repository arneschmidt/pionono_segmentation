import argparse
import numpy as np
from preprocessing_tools.preprocessing_utils import resize_all_images, create_voting_masks, \
    create_crossvalidation_splits, convert_to_rgb, calculate_dataset_statistics

CLASS_COLORS_BGR = [[128, 255, 96], [32, 224, 255], [0, 104, 255], [0, 0, 255], [255, 255, 255]]


parser = argparse.ArgumentParser(description="Preprocess prostate TMA dataset Gleason 2019")
parser.add_argument("--input_dir", "-i", type=str,
                    default="/data/BasesDeDatos/Gleason_2019/original_dataset/",
                    help="Input directory of dataset.")
parser.add_argument("--output_dir", "-o", type=str,
                    default="/data/BasesDeDatos/Gleason_2019/resized_dataset_1024/",
                    help="Output directory of resized images.")
args = parser.parse_args()

config = vars(args)

dataset_specific_names = {'train_img_dir': 'Train_imgs/',
                          'test_img_dir': 'Test_imgs/',
                          'map_dir': 'Maps/',
                          'map_annotator_dirs': ['Maps1_T/', 'Maps2_T/', 'Maps3_T/', 'Maps4_T/', 'Maps5_T/', 'Maps6_T/']}


config.update(dataset_specific_names)

list_gg5 = ['slide001_core145.png', 'slide007_core005.png', 'slide007_core044.png', 'slide003_core068.png',
            'slide007_core016.png', 'slide002_core073.png', 'slide002_core144.png', 'slide001_core010.png',
            'slide002_core009.png', 'slide005_core092.png', 'slide002_core074.png', 'slide002_core140.png',
            'slide002_core143.png', 'slide002_core010.png', 'slide003_core096.png', 'slide007_core043.png']

- 'slide001_core145.png'
- 'slide007_core005.png'
- 'slide007_core044.png'
- 'slide003_core068.png'
- 'slide002_core009.png'
- 'slide005_core092.png'
- 'slide002_core074.png'
- 'slide002_core140.png'
- 'slide002_core143.png'
- 'slide002_core010.png'
- 'slide003_core096.png'
- 'slide007_core043.png'

def convert_masks(mask):
    # The initial classes are 0 (background), 1 (normal tissue), 3 (GG3), 4 (GG4), 5 (GG5), 6 (normal tissue)
    # We move these classes to: 0 (normal tissue), 1 (GG3), 2 (GG4), 3 (GG5), 4 (background)
    ones = np.ones_like(mask)

    mask = mask - 2  # gleason classes are moved to 1,2,3
    mask = np.where(mask == 255, ones * 0, mask)  # normal tissue to 0
    mask = np.where(mask == 4, ones * 0, mask)  # normal tissue to 0
    mask = np.where(mask == 254, ones * 4, mask)  # background  to 4
    return mask

mask_fct = convert_masks

config.update({'resize_resolution': 1024})

resize_all_images(config, config['input_dir'], mask_fct)

create_voting_masks(config, 'majority', dir_name='MV/')

create_voting_masks(config, 'staple', dir_name='STAPLE/')

create_crossvalidation_splits(config, config['output_dir'] + config['train_img_dir'], list_gg5)

convert_to_rgb(config, ['Maps1_T/', 'Maps2_T/', 'Maps3_T/', 'Maps4_T/', 'Maps5_T/', 'Maps6_T/', 'STAPLE/', 'MV/'])

calculate_dataset_statistics(config.output_dir + config.map_dir + 'STAPLE/', 'total')








