import shutil
import torch
import torchvision
import random
from datetime import datetime
import numpy as np
import os
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import functional as AF
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from enum import Enum
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torchvision.transforms import Resize
import cv2
from tqdm import tqdm
from pathlib import Path

from collections import defaultdict
from sklearn.model_selection import train_test_split

from config import settings


class Mode(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


def make_patches(
        source_images: str,
        source_masks: str,
        dest: str,
        size: tuple[int, int],
        shift: tuple[int, int] = None,
):
    if shift is None:
        shift = size

    if not os.path.exists(dest):
        os.mkdir(dest)

    for filename in os.listdir(source_images):
        image_path = os.path.join(source_images, filename)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2GRAY)

        mask_path = os.path.join(source_masks, filename)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_RGB2GRAY)

        height, width = image.shape

        file_name, file_ext = filename.rsplit('.', 1)
        dest_dir = f"{file_name}_{file_ext}_{height}x{width}_{size[0]}x{size[1]}_{shift[0]}x{shift[1]}"
        assert not os.path.exists(os.path.join(dest, dest_dir)), "dest_dir already exist"
        os.mkdir(os.path.join(dest, dest_dir))

        idx = 0
        start_i = 0
        while start_i < height:
            end_i = start_i + size[0]

            start_j = 0
            while start_j < width:
                end_j = start_j + size[1]

                patch_image = image[start_i: min(end_i, height), start_j: min(end_j, width)]
                patch_mask = mask[start_i: min(end_i, height), start_j: min(end_j, width)]

                patch_image = cv2.copyMakeBorder(
                    patch_image, 0, max(end_i - height, 0), 0, max(end_j - width, 0), cv2.BORDER_REFLECT
                )
                patch_mask = cv2.copyMakeBorder(
                    patch_mask, 0, max(end_i - height, 0), 0, max(end_j - width, 0), cv2.BORDER_REFLECT
                )
                # patch = cv2.vconcat([patch_image, patch_mask])
                patch = patch_image

                file_name, file_ext = filename.rsplit(".", 1)
                out_filename = f"{file_name}_{start_i}x{start_j}.{file_ext}"
                cv2.imwrite(os.path.join(dest, dest_dir, out_filename), patch)

                start_j += shift[1]
                idx += 1

            start_i += shift[0]


def copy_and_replace(source_path, destination_path):
    if os.path.exists(destination_path):
        os.remove(destination_path)
    shutil.copy2(source_path, destination_path)


def make_train_val_test(dataset_path: str, train_size: float = 0.6, val_size: float = 0.2):
    filenames = os.listdir(os.path.join(dataset_path, "IMAGES"))
    train_filenames, val_test_filenames = train_test_split(
        filenames, train_size=train_size, shuffle=True, random_state=settings.seed
    )
    val_filenames, test_filenames = train_test_split(
        val_test_filenames, train_size=val_size / (1 - train_size), shuffle=True, random_state=settings.seed
    )

    def _move_images(dir_name: str, filenames: list[str]):
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if not os.path.exists(os.path.join(dir_path, "IMAGES")):
            os.mkdir(os.path.join(dir_path, "IMAGES"))
        if not os.path.exists(os.path.join(dir_path, "MASKS")):
            os.mkdir(os.path.join(dir_path, "MASKS"))

        for filename in filenames:
            in_image_path = os.path.join(dataset_path, "IMAGES", filename)
            in_mask_path = os.path.join(dataset_path, "MASKS", filename)

            out_image_path = os.path.join(dir_path, "IMAGES", filename)
            out_mask_path = os.path.join(dir_path, "MASKS", filename)

            copy_and_replace(in_image_path, out_image_path)
            copy_and_replace(in_mask_path, out_mask_path)

    _move_images("TRAIN", train_filenames)
    _move_images("VAL", val_filenames)
    _move_images("TEST", test_filenames)


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = settings.seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_transform():
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=30, p=0.85),
        A.Normalize(mean=settings.mean, std=settings.std, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return transform


def get_test_transform():
    transform = A.Compose([
        A.Normalize(mean=settings.mean, std=settings.std, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return transform


def calc_mean_std(images_dir: str, mode: str = "RGB"):
    filenames = os.listdir(images_dir)
    images_paths = [os.path.join(images_dir, filename) for filename in filenames]
    images_rgb = list()

    for img in tqdm(images_paths):
        images_rgb.append(np.array(Image.open(img).convert(mode).getdata()) / 255.)

    means = []
    for image_rgb in tqdm(images_rgb):
        means.append(np.mean(image_rgb, axis=0))
    mean = np.mean(means, axis=0)

    variances = []
    for image_rgb in tqdm(images_rgb):
        var = np.mean((image_rgb - mean) ** 2, axis=0)
        variances.append(var)
    std = np.sqrt(np.mean(variances, axis=0))

    return mean, std


def save_weights(checkpoint_callback: ModelCheckpoint):
    best_model_name = Path(checkpoint_callback.best_model_path).name.rsplit(".", 1)[0]
    dir_name = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{best_model_name}"
    dir_name = dir_name.replace(".", "_")
    dir_name = dir_name.replace(":", "-")

    save_dir = os.path.join(settings.experiment_path, dir_name)
    assert not os.path.exists(save_dir)
    os.mkdir(save_dir)

    ckpt_path = os.path.join(save_dir, "weights.ckpt")
    config_path = os.path.join(save_dir, "config.py")
    results_path = os.path.join(save_dir, "results")
    shutil.copy2(checkpoint_callback.best_model_path, ckpt_path)
    shutil.copy2("config.py", config_path)
    os.mkdir(results_path)

    settings.ckpt_dir = dir_name
