import os
import torch
import numpy as np
import cv2
from torch.utils import data

import albumentations as albu
import src.utils.globals as globals
from segmentation_models_pytorch.encoders import get_preprocessing_fn


def get_training_augmentation():

    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


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
        self.class_values = list(range(self.class_no))
        if globals.config['data']['ignore_last_class']:
            self.class_values.append(self.class_no)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        class_values = self.class_values.copy()

        # Delete this part after classes have been changed by Miguel
        # move th class 0 to the last dimension such that network and labels have the same indices
        if globals.config['data']['ignore_last_class']:
            delete_value = np.ones_like(mask) * self.class_no
            mask = mask - 1
            mask = np.where(mask==-1, delete_value, mask)

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


def get_data_supervised():
    config = globals.config
    batch_size = config['model']['batch_size']

    train_image_folder = os.path.join(config['data']['path'], config['data']['train']['images'])
    train_label_folder = os.path.join(config['data']['path'], config['data']['train']['masks'])
    val_image_folder = os.path.join(config['data']['path'], config['data']['val']['images'])
    val_label_folder = os.path.join(config['data']['path'], config['data']['val']['masks'])
    test_image_folder = os.path.join(config['data']['path'], config['data']['test']['images'])
    test_label_folder = os.path.join(config['data']['path'], config['data']['test']['masks'])

    encoder_name = config['model']['encoder']['backbone']
    encoder_weights = config['model']['encoder']['weights']
    preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)

    train_dataset = CustomDataset(train_image_folder, train_label_folder, augmentation=get_training_augmentation(),
                                  preprocessing = get_preprocessing(preprocessing_fn))
    validate_dataset = CustomDataset(val_image_folder, val_label_folder,
                                     preprocessing = get_preprocessing(preprocessing_fn))
    test_dataset = CustomDataset(test_image_folder, test_label_folder,
                                 preprocessing = get_preprocessing(preprocessing_fn))

    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                  drop_last=True)
    validateloader = data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=batch_size, drop_last=False)
    testloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=batch_size,
                                 drop_last=False)  # batch_size to 1 for the visualizing images

    return trainloader, validateloader, testloader