import os
import torch
import numpy as np
import cv2
import functools

from torch.utils import data

import albumentations as albu
import utils.globals as globals
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from utils.preprocessing import get_preprocessing_fn_without_normalization


def get_training_augmentation():
    aug_config = globals.config['data']['augmentation']
    if aug_config['use_augmentation']:
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),

            albu.Blur(blur_limit=aug_config['gaussian_blur_kernel'], p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=aug_config['brightness_limit'],
                                          contrast_limit=aug_config['contrast_limit'],
                                          p=0.5),
            albu.HueSaturationValue(hue_shift_limit=aug_config['hue_shift_limit'],
                                    sat_shift_limit=aug_config['sat_shift_limit'],
                                    p=0.5)
        ]
        composed_transform = albu.Compose(train_transform)
    else:
        composed_transform = None
    return composed_transform


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

# def my_preprocess_input(x):
#
#     x = x / 255.0
#
#     return x
#
# def get_my_prec():
#     return functools.partial(my_preprocess_input)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

# =============================================

class CustomDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_no = globals.config['data']['class_no']
        self.class_values = self.set_class_values(self.class_no)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, self.ids[i]

    def __len__(self):
        return len(self.ids)

    def set_class_values(self, class_no):
        if globals.config['data']['ignore_last_class']:
            class_values = list(range(class_no + 1))
        else:
            class_values = list(range(class_no))
        return class_values


def get_data_supervised():
    config = globals.config
    batch_size = config['model']['batch_size']
    normalization = config['data']['normalization']

    train_image_folder = os.path.join(config['data']['path'], config['data']['train']['images'])
    train_label_folder = os.path.join(config['data']['path'], config['data']['train']['masks'])
    val_image_folder = os.path.join(config['data']['path'], config['data']['val']['images'])
    val_label_folder = os.path.join(config['data']['path'], config['data']['val']['masks'])
    test_image_folder = os.path.join(config['data']['path'], config['data']['test']['images'])
    test_label_folder = os.path.join(config['data']['path'], config['data']['test']['masks'])

    if normalization:
        encoder_name = config['model']['encoder']['backbone']
        encoder_weights = config['model']['encoder']['weights']
        preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)
    else:
        preprocessing_fn = get_preprocessing_fn_without_normalization()

    preprocessing = get_preprocessing(preprocessing_fn)

    train_dataset = CustomDataset(train_image_folder, train_label_folder, augmentation=get_training_augmentation(),
                                  preprocessing = preprocessing)
    validate_dataset = CustomDataset(val_image_folder, val_label_folder, preprocessing = preprocessing)
    test_dataset = CustomDataset(test_image_folder, test_label_folder, preprocessing = preprocessing)

    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    validateloader = data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size,
                                     drop_last=False)
    testloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size,
                                 drop_last=False)

    return trainloader, validateloader, testloader