import argparse
import os
import shutil
import yaml

parser = argparse.ArgumentParser(
    description="This code copies the validation masks of each annotator that are used for visualization")
parser.add_argument("--input_dir", "-i", type=str,
                    default="/home/arne/datasets/Gleason_2019/resized_dataset_1024/",
                    help="Input directory of dataset.")
parser.add_argument("--output_dir", "-o", type=str,
                    default="/home/arne/datasets/Gleason_2019/resized_dataset_1024/rgb_images/common_masks/",
                    help="Output directory of validation masks.")
args = parser.parse_args()

train_img_dir = 'Train_imgs/'
test_img_dir = 'Test_imgs/'
map_annotator_dirs = ['Maps1_T/', 'Maps2_T/', 'Maps3_T/', 'Maps4_T/', 'Maps5_T/', 'Maps6_T/']
rgg_masks_dir = 'rgb_images/'

with open(args.dataset_config) as file:
    config = yaml.full_load(file)

masks = config['data']['visualize_images']['val']

for a in range(len(map_annotator_dirs)):
    map_annotator_dir = map_annotator_dirs[a]
    print(map_annotator_dir)
    for m in range(len(masks)):
        mask_name = masks[m]

        mask_path_in = args.input_dir + rgg_masks_dir + map_annotator_dir + mask_name
        if os.path.exists(mask_path_in):
            dir_path_out = args.output_dir + map_annotator_dir
            os.makedirs(dir_path_out, exist_ok=True)
            mask_path_out = dir_path_out + mask_name
            shutil.copy(mask_path_in, mask_path_out)

