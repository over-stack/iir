import os
import copy

from models.BaseConvNet import BaseConvNet
from models.IIRConvNet import IIRConvNet
from train import train_model, test_model
from utils import make_patches, set_deterministic, make_train_val_test, calc_mean_std
from config import settings


def main():
    set_deterministic(settings.seed)
    # model = BaseConvNet(settings.in_channels, settings.num_filters, 3, 1)
    model = IIRConvNet(settings.in_channels, settings.num_filters, 3, 1)
    # train_model(copy.deepcopy(model))
    test_model(model, settings.ckpt_path)


if __name__ == "__main__":
    main()
