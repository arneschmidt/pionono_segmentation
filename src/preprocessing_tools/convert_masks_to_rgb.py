import argparse
import os
import cv2
import shutil
import random
import numpy as np
from scipy import stats
from utils.saving import CLASS_COLORS_BGR


parser = argparse.ArgumentParser(description="Resize Images of prostate TMA")
parser.add_argument("--input_dir", "-i", type=str,
                    default="/data/BasesDeDatos/Gleason_2019/resized_dataset_1024/",
                    help="Input directory of dataset.")
parser.add_argument("--output_dir", "-o", type=str,
                    default="/data/BasesDeDatos/Gleason_2019/resized_dataset_1024/rgb_masks/",
                    help="Output directory of converted masks.")
args = parser.parse_args()

map_dir = 'Maps/'
# map_annotator_dirs = ['Maps1_T/', 'Maps2_T/', 'Maps3_T/', 'Maps4_T/', 'Maps5_T/', 'Maps6_T/', 'STAPLE/', 'MV/']
map_annotator_dirs = ['STAPLE/', 'MV/']


os.makedirs(args.output_dir, exist_ok=True)
for m in range(len(map_annotator_dirs)):
    map_annotator_dir = map_annotator_dirs[m]
    in_dir = args.input_dir + map_dir + map_annotator_dir
    out_dir = args.output_dir + map_dir + map_annotator_dir
    os.makedirs(out_dir, exist_ok=True)
    img_file_list = os.listdir(in_dir)
    print(map_annotator_dir)
    print('Images found:' + str(len(img_file_list)))
    resolution_list = []
    for img_file in img_file_list:
        img_path_in = in_dir + img_file
        img_path_out = out_dir + img_file

        image = cv2.imread(img_path_in)
        print('Image: ' + img_file + ' Shape: ' + str(image.shape))

        # classes : 0 (normal tissue), 1 (GG3), 2 (GG4), 3 (GG5), 4 (background)
        ones = np.ones_like(image)

        for c in range(len(CLASS_COLORS_BGR)):
            image = np.where(image==c, ones*CLASS_COLORS_BGR[c], image)

        assert np.all(image >= 0)
        assert np.all(image <= 255)

        cv2.imwrite(img_path_out, image)