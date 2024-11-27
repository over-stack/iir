"""import os

dir_ = "datasets/dibco2017_training_pack/train+val_patches/validation/images/printed"

for filename in os.listdir(dir_):
    source = os.path.join(dir_, filename)
    dest = os.path.join(dir_, f"p_{filename}")

    os.rename(source, dest)
"""

import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import settings
from utils import make_patches

make_patches(
    "datasets/DIBCO2017/IMAGES",
    "datasets/DIBCO2017/MASKS",
    settings.test_images_path,
    (settings.height, settings.width),
    (settings.y_shift, settings.x_shift),
)

"""
def make_patches(
        source: str,
        dest: str,
        size: tuple[int, int],
        shift: tuple[int, int] = None,
):
    if shift is None:
        shift = size

    if not os.path.exists(dest):
        os.mkdir(dest)

    def _make_patches(dir_name: str, to_gray: bool = False):
        in_dir_path = os.path.join(dataset_path, dir_name)
        out_dir_path = os.path.join(output_path, dir_name)

        if not os.path.exists(out_dir_path):
            os.mkdir(out_dir_path)

        filenames = os.listdir(in_dir_path)

        for filename in filenames:
            image_path = os.path.join(in_dir_path, filename)

            image = cv2.imread(image_path)
            assert len(image.shape) == 3, "Image shape supposed to be 3dim"

            height, width, _ = image.shape
            if to_gray:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            idx = 0
            start_i = 0
            while start_i < height:
                end_i = start_i + size[0]

                start_j = 0
                while start_j < width:
                    end_j = start_j + size[1]

                    patch = image[start_i: min(end_i, height), start_j: min(end_j, width)]
                    patch = cv2.copyMakeBorder(
                        patch, 0, max(end_i - height, 0), 0, max(end_j - width, 0), cv2.BORDER_REFLECT
                    )

                    file_name, file_ext = filename.rsplit(".", 1)
                    out_filename = f"{file_name}_{idx}.{file_ext}"
                    cv2.imwrite(os.path.join(out_dir_path, out_filename), patch)

                    start_j += size[1] - overlapping[1]
                    idx += 1

                start_i += size[0] - overlapping[0]

    _make_patches("IMAGES")
    _make_patches("MASKS", to_gray=True)
"""