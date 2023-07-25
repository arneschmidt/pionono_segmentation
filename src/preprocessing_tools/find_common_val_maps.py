import argparse
import os
import shutil
import yaml
import numpy as np

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
rgb_masks_dir = 'rgb_images/'
crossval_dirs = ['Crossval0/', 'Crossval1/', 'Crossval2/', 'Crossval3/']

os.makedirs(args.output_dir, exist_ok=True)

intersect_maps = np.array([])
# Create list of all common masks
for a in range(len(map_annotator_dirs)):
    map_annotator_dir = map_annotator_dirs[a]
    mask_path_dir = args.input_dir + rgb_masks_dir + map_annotator_dir
    a_maps = np.array(os.listdir(mask_path_dir))
    if a == 0: # init as all maps of first annotator
        intersect_maps = a_maps
    else:
        intersect_maps = np.intersect1d(a_maps, intersect_maps)

# create list for logging for all crossvalid splits
for c in range(len(crossval_dirs)):
    crossval_dir = args.input_dir + crossval_dirs[c] + 'val/'
    c_maps = np.array(os.listdir(crossval_dir))
    c_intersect_maps = np.intersect1d(c_maps, intersect_maps)

    c_out_dir = args.output_dir + crossval_dirs[c]
    os.makedirs(c_out_dir, exist_ok=True)
    with open(c_out_dir + '/validation_images.txt', 'w') as f:
        for m in range(len(c_intersect_maps)):
            print("- '" + c_intersect_maps[m] + "'", file=f)
            for a in range(len(map_annotator_dirs)):
                mask_path_in = args.input_dir + rgb_masks_dir + map_annotator_dirs[a] + c_intersect_maps[m]
                if os.path.exists(mask_path_in):
                    dir_path_out = args.output_dir + crossval_dirs[c] + map_annotator_dirs[a]
                    os.makedirs(dir_path_out, exist_ok=True)
                    mask_path_out = dir_path_out + c_intersect_maps[m]
                    shutil.copy(mask_path_in, mask_path_out)



