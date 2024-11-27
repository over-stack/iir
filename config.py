import os.path
from pathlib import Path
import datetime

import torch


class Settings:
    gpu_available = torch.cuda.is_available()
    accelerator = 'gpu' if gpu_available else 'cpu'
    device = 'cuda' if gpu_available else 'cpu'
    pin_memory = gpu_available
    log_every_n_steps = 5
    eps = 1e-9

    metrics_thresholds = [0.5]
    n_cols, n_rows = 5, 2

    threshold = 0.5

    tran_images_path = "datasets/dibco2017_training_pack/patches/train/images/all"
    val_images_path = "datasets/dibco2017_training_pack/patches/validation/images/all"
    test_images_path = "datasets/dibco2017_training_pack/patches/test_images"
    test_masks_path = "datasets/dibco2017_training_pack/patches/test_masks"
    test_images_full_path = "datasets/DIBCO2017/IMAGES"

    weights_dir = "weights"
    experiment_name = "iir_first_experiment"
    ckpt_dir = "2024-11-26 14-30-00_epoch=0 val_acc_epoch=0_98 val_f1score_epoch=0_99 val_iou_epoch=0_94"

    # mean = [0.79127279, 0.73286374, 0.62693299]
    # std = [0.15295595, 0.17590371, 0.1758023]

    mean = [0.8351939924978029]
    std = [0.24232973282031978]

    height, width = 128, 128
    y_shift, x_shift = 96, 96
    num_workers = 4

    in_channels = 1
    num_classes = 1
    num_filters = 24

    num_epochs = 10
    batch_size = 48

    lr = 1e-2
    weight_decay = 1e-7
    lr_reduce_factor = 0.5
    lr_reduce_patience = 2
    early_stopping_patience = 3
    seed = 4

    def __init__(self):
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)

        if not os.path.exists(self.experiment_path):
            os.mkdir(self.experiment_path)

    @property
    def experiment_path(self):
        return os.path.join(self.weights_dir, self.experiment_name)

    @property
    def ckpt_path(self):
        return os.path.join(self.experiment_path, self.ckpt_dir, "weights.ckpt")

    @property
    def config_path(self):
        return os.path.join(self.experiment_path, self.ckpt_dir, "config.py")

    @property
    def results_path(self):
        return os.path.join(self.experiment_path, self.ckpt_dir, "results")


settings = Settings()
