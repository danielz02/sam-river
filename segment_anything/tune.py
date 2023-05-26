import os
import argparse
import sys
from collections import defaultdict, deque
import pickle
from collections.abc import Iterable
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import cv2


import torch
from param import Dict
from skimage.morphology import binary_dilation
from torch import Tensor
from torchgeo.datasets.utils import _list_dict_to_dict_list

from segment_anything.data.naip import RiverBank

from segment_anything.data.dem import NAIP
from segment_anything.modeling.common import LayerNorm2d
from segment_anything.utils.transforms import ResizeLongestSide

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchmetrics.functional import dice
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import segmentation_models_pytorch as smp
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler
from torchgeo.datasets import RasterDataset, VectorDataset

from torchvision.ops import sigmoid_focal_loss

# Add the SAM directory to the system path
sys.path.append("./segment-anything")
from segment_anything import sam_model_registry
from skimage import measure

NUM_WORKERS = 0  # https://github.com/pytorch/pytorch/issues/42518
NUM_GPUS = torch.cuda.device_count()
DEVICE = 'cuda'


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def stack_samples(samples: Iterable[dict[Any, Any]]) -> dict[Any, list[Any]]:
    """Stack a list of samples along a new axis.

    Useful for forming a mini-batch of samples to pass to
    :class:`torch.utils.data.DataLoader`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated = _list_dict_to_dict_list(samples)
    deleted = []
    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            collated[key] = torch.stack(value)
        else:
            deleted.append(key)
    for key in deleted:
        del collated[key]
    return collated


def random_poi_from_mask(mask_image, num_points=10):
    h, w = mask_image.shape
    connected_regions = measure.label(mask_image)
    u, count = np.unique(connected_regions[mask_image], return_counts=True, )
    y_pts, x_pts = np.where(np.logical_and(mask_image, connected_regions == u[np.argmax(count)]))

    idx = np.random.choice(np.arange(len(y_pts)), size=num_points)

    return np.stack([x_pts[idx], y_pts[idx]], -1)


class SAMFinetuner(pl.LightningModule):

    def __init__(
            self,
            model_type,
            checkpoint_path,
            image_size,
            freeze_image_encoder=False,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            batch_size=1,
            learning_rate=1e-4,
            weight_decay=1e-4,
            metrics_interval=10,
    ):
        super(SAMFinetuner, self).__init__()

        self.model_type = model_type
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.mask_in_chans = 16
        self.prompt_adapter = nn.Sequential(
            nn.Conv2d(1, self.mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(self.mask_in_chans // 4, self.mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(self.mask_in_chans),
            nn.GELU(),
            nn.Conv2d(self.mask_in_chans, 1, kernel_size=1),
        )
        self.image_size = image_size
        self.freeze_image_encoder = freeze_image_encoder
        self.resize_transform = ResizeLongestSide(self.model.img_size)
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))

        self.metrics_interval = metrics_interval

    def forward(self, imgs, bboxes, labels):
        _, c, h, w = imgs.shape
        # fix: change from [:, :3, h, w] to [:, :3, :, :]
        imgs, dems = imgs[:, :3, :, :], imgs[:, -1, :, :].unsqueeze(1)

        # r, g, b, nir = imgs[:, :4, h, w].transpose(1, 0, 2, 3)
        # ndvi_mask = (nir - r) / (nir + r) > 0.3
        # ndwi_mask = (g - nir) / (g + nir) > 0
        # ndvi_mask[ndwi_mask] = False
        #
        # ndwi_mask_dilation = binary_dilation(ndwi_mask)
        # ndwi_mask_dilation[ndwi_mask] = False

        features = self.model.image_encoder(imgs)
        dense_prompts = self.prompt_adapter(dems)
        # num_masks = sum([len(b) for b in bboxes])

        loss_focal = loss_dice = loss_iou = 0.
        predictions = []
        tp, fp, fn, tn = [], [], [], []
        for feature, bbox, dense_prompt, label in zip(features, bboxes, dense_prompts, labels):
            # pos_pts = np.concatenate([
            #     random_poi_from_mask(ndwi_mask, num_points=5),
            #     random_poi_from_mask(ndwi_mask_dilation, num_points=10)
            # ])
            # neg_pts = random_poi_from_mask(ndvi_mask, num_points=10)
            # poi_labels = np.concatenate([np.ones(len(pos_pts)), np.zeros(len(neg_pts))])
            # pois = np.concatenate([pos_pts, neg_pts])
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,  # (pois, poi_labels),
                boxes=None,
                masks=dense_prompt.unsqueeze(0),
            )
            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=feature.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # Upscale the masks to the original image resolution
            masks = F.interpolate(
                low_res_masks,
                (h, w),
                mode="bilinear",
                align_corners=False,
            )
            predictions.append(masks)
            # Compute the iou between the predicted masks and the ground truth masks
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                masks,
                label.unsqueeze(1),
                mode='binary',
                threshold=0.5,
            )
            batch_iou = smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn)
            # Compute the loss
            masks = masks.squeeze(1).flatten(1)
            label = label.flatten(1)
            loss_focal += sigmoid_focal_loss(masks, label.float())  # / num_masks
            loss_dice += dice(preds=masks, target=label.float())  # / num_masks
            loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='sum')  # / num_masks
            tp.append(batch_tp)
            fp.append(batch_fp)
            fn.append(batch_fn)
            tn.append(batch_tn)
        return {
            'loss': 20. * loss_focal + loss_dice + loss_iou,  # SAM default loss
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_iou': loss_iou,
            'predictions': predictions,
            'tp': torch.cat(tp),
            'fp': torch.cat(fp),
            'fn': torch.cat(fn),
            'tn': torch.cat(tn),
        }

    def training_step(self, batch, batch_nb):
        # fix: images -> image
        imgs, bboxes, labels = batch["image"], batch["bbox"], batch[""]
        outputs = self.forward(imgs, bboxes, labels)

        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.train_metric[metric].append(outputs[metric])

        # aggregate step metrics
        step_metrics = [torch.cat(list(self.train_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        metrics = {
            "loss": outputs["loss"],
            "loss_focal": outputs["loss_focal"],
            "loss_dice": outputs["loss_dice"],
            "loss_iou": outputs["loss_iou"],
            "train_per_mask_iou": per_mask_iou,
        }
        self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
        return metrics

    def validation_step(self, batch, batch_nb):
        imgs, bboxes, labels = batch
        outputs = self(imgs, bboxes, labels)
        outputs.pop("predictions")
        return outputs

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if NUM_GPUS > 1:
            outputs = all_gather(outputs)
            # the outputs are a list of lists, so flatten it
            outputs = [item for sublist in outputs for item in sublist]
        # aggregate step metrics
        step_metrics = [
            torch.cat(list([x[metric].to(self.device) for x in outputs]))
            for metric in ['tp', 'fp', 'fn', 'tn']
        ]
        # per mask IoU means that we first calculate IoU score for each mask
        # and then compute mean over these scores
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")

        metrics = {"val_per_mask_iou": per_mask_iou}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        def warmup_step_lr_builder(warmup_steps, milestones, gamma):
            def warmup_step_lr(steps):
                if steps < warmup_steps:
                    lr_scale = (steps + 1.) / float(warmup_steps)
                else:
                    lr_scale = 1.
                    for milestone in sorted(milestones):
                        if steps >= milestone * self.trainer.estimated_stepping_batches:
                            lr_scale *= gamma
                return lr_scale

            return warmup_step_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            warmup_step_lr_builder(250, [0.66667, 0.86666], 0.1)
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }


class RiverbankDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, img_size: int = 512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str):
        naip_train = NAIP(root="/projects/bbkc/danielz/NAIP/Leaf-Off-2011/ROI/2011_Leaf_off/Mackinaw_Main/train/")
        naip_val = NAIP(root="/projects/bbkc/danielz/NAIP/Leaf-Off-2011/ROI/2011_Leaf_off/Mackinaw_Main/val/")
        dem_dataset = RasterDataset(
            root="/projects/bbkc/danielz/ILHMP/2010s/DTM_ROI/Mackinaw_River_Stream_Order_Buffer/"
        )
        mackinaw_river = RiverBank(root="/projects/bbkc/danielz/river_bank/data/Shapefiles/Mackinaw_2011/")
        self.train_dataset = (naip_train & mackinaw_river) & dem_dataset
        self.val_dataset = (naip_val & mackinaw_river) & dem_dataset

    def train_dataloader(self):
        sampler = RandomBatchGeoSampler(self.train_dataset, size=self.img_size, batch_size=self.batch_size)
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=stack_samples,
            batch_sampler=sampler,
            num_workers=NUM_WORKERS,
        )
        return train_loader

    def val_dataloader(self):
        sampler = GridGeoSampler(self.train_dataset, size=self.img_size, stride=(self.img_size // 2))
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=stack_samples,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            sampler=sampler,
            shuffle=False
        )
        return val_loader

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, help="model type", choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to the checkpoint")
    parser.add_argument("--freeze_image_encoder", action="store_true", help="freeze image encoder")
    parser.add_argument("--freeze_prompt_encoder", action="store_true", help="freeze prompt encoder")
    parser.add_argument("--freeze_mask_decoder", action="store_true", help="freeze mask decoder")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--steps", type=int, default=1500, help="number of steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--metrics_interval", type=int, default=50, help="interval for logging metrics")
    parser.add_argument("--output_dir", type=str, default=".", help="path to save the model")

    torch.set_float32_matmul_precision("medium")

    args = parser.parse_args()
    # create the model
    model = SAMFinetuner(
        args.model_type,
        args.checkpoint_path,
        image_size=args.image_size,
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_prompt_encoder=args.freeze_prompt_encoder,
        freeze_mask_decoder=args.freeze_mask_decoder,

        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        metrics_interval=args.metrics_interval,
    )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{step}-{val_per_mask_iou:.2f}',
            save_last=True,
            save_top_k=1,
            monitor="val_per_mask_iou",
            mode="max",
            save_weights_only=True,
            every_n_train_steps=args.metrics_interval,
        ),
    ]
    trainer = pl.Trainer(
        strategy='ddp' if NUM_GPUS > 1 else "auto",
        accelerator=DEVICE,
        devices=NUM_GPUS,
        precision=32,
        callbacks=callbacks,
        max_epochs=-1,
        max_steps=args.steps,
        val_check_interval=args.metrics_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )

    datamodule = RiverbankDataModule()

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
