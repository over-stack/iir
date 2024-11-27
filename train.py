import os
import math
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm

from natsort import natsorted
import numpy as np
from PIL import Image
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchmetrics

from config import settings
from utils import set_deterministic, get_train_transform, get_test_transform, Mode, seed_worker, save_weights
from drd import drd
from dataset import DIBCO2017Dataset


class TrainingModule(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.loss = nn.BCEWithLogitsLoss()

        self.train_accuracy = torchmetrics.Accuracy("binary", settings.threshold)
        self.train_precision = torchmetrics.Precision("binary", settings.threshold)
        self.train_recall = torchmetrics.Recall("binary", settings.threshold)
        self.train_f1score = torchmetrics.F1Score("binary", settings.threshold)
        self.train_iou = torchmetrics.JaccardIndex("binary", settings.threshold)

        self.val_accuracy = torchmetrics.Accuracy("binary", settings.threshold)
        self.val_precision = torchmetrics.Precision("binary", settings.threshold)
        self.val_recall = torchmetrics.Recall("binary", settings.threshold)
        self.val_f1score = torchmetrics.F1Score("binary", settings.threshold)
        self.val_iou = torchmetrics.JaccardIndex("binary", settings.threshold)

        self.thresh = nn.Threshold(threshold=settings.threshold, value=0)

        self.flag = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        prediction = self.model(x)
        loss = self.loss(prediction, y)

        self.train_accuracy(prediction, y)
        self.train_precision(prediction, y)
        self.train_recall(prediction, y)
        self.train_f1score(prediction, y)
        self.train_iou(prediction, y)

        metrics = {
            "train_loss": loss,
            "train_acc": self.train_accuracy,
            "train_precision": self.train_precision,
            "train_recall": self.train_recall,
            "train_f1score": self.train_f1score,
            "train_iou": self.train_iou,
        }

        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=False)
        self.flag = True

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        prediction = self.model(x)
        loss = self.loss(prediction, y)

        self.val_accuracy(prediction, y)
        self.val_precision(prediction, y)
        self.val_recall(prediction, y)
        self.val_f1score(prediction, y)

        metrics = {
            "val_loss": loss,
            "val_acc": self.val_accuracy,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "val_f1score": self.val_f1score,
            "val_iou": self.train_iou,
        }
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=False)

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch

        prediction = self.model(x)
        prediction = nn.Sigmoid()(prediction)
        prediction = self.thresh(prediction)

        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=settings.lr_reduce_factor,
            patience=settings.lr_reduce_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss_epoch",
                "frequency": 1,
            },
        }


def train_model(model):
    if settings.seed is not None:
        set_deterministic(settings.seed)

    train_transform = get_train_transform()
    test_transform = get_test_transform()

    train_dataset = DIBCO2017Dataset(Mode.TRAIN, settings.tran_images_path, train_transform)
    val_dataset = DIBCO2017Dataset(Mode.VAL, settings.val_images_path, test_transform)

    g = torch.Generator()
    g.manual_seed(settings.seed)

    train_dataloader = DataLoader(
        train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers,
        pin_memory=settings.pin_memory, persistent_workers=True, worker_init_fn=seed_worker, generator=g
    )
    validation_dataloader = DataLoader(
        val_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers,
        pin_memory=settings.pin_memory, persistent_workers=True, worker_init_fn=seed_worker, generator=g
    )

    earlystopping_callback = EarlyStopping(
        monitor='val_loss_epoch',
        mode='min',
        patience=settings.early_stopping_patience,
        verbose=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'runs/{settings.experiment_name}',
        filename='{epoch} {val_acc_epoch:.2f} {val_f1score_epoch:.2f} {val_iou_epoch:.2f}',
        monitor='val_loss_epoch',
        mode='min',
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=settings.num_epochs,
        accelerator=settings.accelerator,
        devices=1,
        callbacks=[earlystopping_callback, checkpoint_callback],
        log_every_n_steps=settings.log_every_n_steps,
        enable_checkpointing=True,
        logger=True,
    )

    training_module = TrainingModule(model)
    trainer.fit(training_module, train_dataloader, validation_dataloader)

    save_weights(checkpoint_callback)

    return training_module.model


def test_model(model, ckpt_path):
    model = TrainingModule.load_from_checkpoint(ckpt_path, model=model)

    accuracy = torchmetrics.Accuracy("binary", settings.threshold)
    precision = torchmetrics.Precision("binary", settings.threshold)
    recall = torchmetrics.Recall("binary", settings.threshold)
    f1score = torchmetrics.F1Score("binary", settings.threshold)
    iou = torchmetrics.JaccardIndex("binary", settings.threshold)

    g = torch.Generator()
    g.manual_seed(settings.seed)
    test_transform = get_test_transform()
    filenames = sorted(os.listdir(settings.test_images_path))

    metrics = defaultdict(list)

    for filename in natsorted(filenames):
        isz_psz_psh = filename.rsplit("_")
        file_name = isz_psz_psh[0]
        file_ext = isz_psz_psh[1]
        image_size = list(map(int, isz_psz_psh[2].split("x")))
        patch_size = list(map(int, isz_psz_psh[3].split("x")))
        patch_shift = list(map(int, isz_psz_psh[4].split("x")))

        if not all([
            patch_size[0] == settings.height, patch_size[1] == settings.width,
            patch_shift[0] == settings.y_shift, patch_shift[1] == settings.x_shift
        ]):
            continue  # choose only parameters equals from config

        images_path = os.path.join(settings.test_images_path, filename)
        test_dataset = DIBCO2017Dataset(Mode.TEST, images_path, test_transform)
        test_dataloader = DataLoader(
            test_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers,
            pin_memory=settings.pin_memory, persistent_workers=True, worker_init_fn=seed_worker, generator=g
        )

        patches_count = (math.ceil(image_size[0] / patch_shift[0]), math.ceil(image_size[1] / patch_shift[1]))
        result_size = (
            (patches_count[0] - 1) * patch_shift[0] + patch_size[0],
            (patches_count[1] - 1) * patch_shift[1] + patch_size[1],
        )
        assert result_size[0] >= 0 and result_size[1] >= 0

        result = torch.zeros(result_size, dtype=torch.float32, device="cpu")
        coefs = torch.zeros(result_size, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            idx = 0
            for batch in test_dataloader:
                batch = batch.to(settings.device)
                x = batch
                prediction = model(x)
                prediction = nn.Sigmoid()(prediction)  # .squeeze(1).detach().to("cpu").numpy()
                for k in range(prediction.shape[0]):
                    i = idx // patches_count[1]
                    j = idx - i * patches_count[1]

                    result[
                        i * patch_shift[0]: i * patch_shift[0] + patch_size[0],
                        j * patch_shift[1]: j * patch_shift[1] + patch_size[1],
                    ] += prediction[k][0].detach().to("cpu")
                    coefs[
                        i * patch_shift[0]: i * patch_shift[0] + patch_size[0],
                        j * patch_shift[1]: j * patch_shift[1] + patch_size[1],
                    ] += 1

                    idx += 1

        result /= coefs
        result = (result > settings.threshold)[:image_size[0], :image_size[1]].to(torch.int64).unsqueeze(0).unsqueeze(0)
        result_np = result[0, 0].numpy().astype(np.uint8)

        mask_path = os.path.join(settings.test_masks_path, f"{file_name}.{file_ext}")
        mask_np = np.array(Image.open(mask_path).convert("L")) / 255.
        mask_np = mask_np.astype(np.uint8)
        mask_torch = torch.from_numpy(mask_np.astype(np.int64)).unsqueeze(0).unsqueeze(0)

        image_full_path = os.path.join(settings.test_images_full_path, f"{file_name}.{file_ext}")
        image_full = np.array(Image.open(image_full_path).convert("L"))

        metrics["accuracy"].append(accuracy(result, mask_torch))
        metrics["precision"].append(precision(result, mask_torch))
        metrics["recall"].append(recall(result, mask_torch))
        metrics["f1score"].append(f1score(result, mask_torch))
        metrics["iou"].append(iou(result, mask_torch))
        metrics["drd"].append(drd(mask_np.astype(np.int32), result_np.astype(np.int32)))

        print(f"filename: {file_name}")
        print("\t".join([f"{key}: {value[-1]:.3f}" for key, value in metrics.items()]))

        """fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image_full)
        ax[1].imshow(result_np * 255)
        ax[2].imshow(mask_np * 255)
        plt.show()"""

        Image.fromarray(
            np.concatenate([
                image_full,
                result_np * 255,
                mask_np * 255,
            ], axis=1),
        ).save(os.path.join(settings.results_path, f"{file_name}.{file_ext}"))

    print("MEAN METRICS")
    print("\t".join([f"{key}: {sum(value) / len(value):.3f}" for key, value in metrics.items()]))
