import os
from glob import glob

import matplotlib.pyplot as plt
from typing import Any, Dict, Optional

import numpy as np
import rasterio
import torch
from rasterio.coords import BoundingBox
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
import torch.nn.functional as F
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.datasets import VectorDataset, RasterDataset, stack_samples, unbind_samples

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class NAIP(RasterDataset):
    """National Agriculture Imagery Program (NAIP) dataset.

    The `National Agriculture Imagery Program (NAIP)
    <https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/>`_
    acquires aerial imagery during the agricultural growing seasons in the continental
    U.S. A primary goal of the NAIP program is to make digital ortho photography
    available to governmental agencies and the public within a year of acquisition.

    NAIP is administered by the USDA's Farm Service Agency (FSA) through the Aerial
    Photography Field Office in Salt Lake City. This "leaf-on" imagery is used as a base
    layer for GIS programs in FSA's County Service Centers, and is used to maintain the
    Common Land Unit (CLU) boundaries.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.fisheries.noaa.gov/inport/item/49508/citation
    """

    filename_glob = "*_ortho_1-1_h*_s_*_*.tif"
    filename_regex = r"(?P<fid>\d+)_ortho_1-1_h(?P<color>[a-z])_s_(?P<state>[a-z]+)(?P<fips>\d+)_" \
                     r"(?P<yyyy>\d+)_..*"
    # filename_glob = "*_*.tif"
    # filename_regex = r"(?P<fid>\d+)_(?P<tid1>\d+)(?P<tid2>[a-z]+)(?P<tid3>\d+).tif"
    # filename_regex = r"patch_(?P<x>\d+)_(?P<y>\d+).tif"

    # Plotting
    all_bands = ["R", "G", "B", "NIR"]
    rgb_bands = ["R", "G", "B"]

    def plot(self, sample: Dict[str, Any], show_titles: bool = True, suptitle: Optional[str] = None) -> plt.Figure:
        """Plot a sample from the dataset.

            Args:
                sample: a sample returned by :meth:`RasterDataset.__getitem__`
                show_titles: flag indicating whether to show titles above each panel
                suptitle: optional string to use as a suptitle

            Returns:
                a matplotlib Figure with the rendered sample

            .. versionchanged:: 0.3
               Method now takes a sample dict, not a Tensor. Additionally, possible to
               show subplot titles and/or use a custom suptitle.
        """
        image = sample["image"][0:3, :, :].permute(1, 2, 0).clone()

        if "mask" in sample:
            mask = sample["mask"]
            image[mask, :] = torch.tensor([255, 0, 0], device=image.device, dtype=image.dtype)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        ax.imshow(image.cpu().numpy())
        ax.axis("off")
        if "point_coords" in sample:
            if isinstance(sample["point_coords"], torch.Tensor):
                pts = sample["point_coords"].cpu().numpy()
            else:
                pts = sample["point_coords"]
            ax.scatter(x=pts[0, :, 0], y=pts[0, :, 1])
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class RiverBank(VectorDataset):
    # filename_glob = "*_Stream_Order.shp.gpkg"
    # filename_glob = "patch_*.shp"
    filename_glob = "2011_Leaf_off.shp"


def mask2points(masks: torch.LongTensor):
    b, h, w = masks.shape
    b_idx, h_idx, w_idx = torch.nonzero(masks, as_tuple=True)
    assert torch.all(h_idx < h) and torch.all(w_idx < w)
    _, pts_counts = torch.unique(b_idx, return_counts=True)
    h_idx = h_idx.split(pts_counts.tolist())
    w_idx = w_idx.split(pts_counts.tolist())

    point_prompts = [torch.stack([x, y], dim=-1)[range(0, len(x), 20)].unsqueeze(dim=0) for x, y in zip(w_idx, h_idx)]

    return point_prompts


def poi_from_mask(mask_image, num_points=10, how="center", deviate=0):
    from skimage import measure

    h, w = mask_image.shape
    connected_regions = measure.label(mask_image)
    u, count = np.unique(connected_regions[mask_image], return_counts=True,)
    y_pts, x_pts = np.where(np.logical_and(mask_image, connected_regions == u[np.argmax(count)]))

    x_unique = np.unique(x_pts)
    x_unique = x_unique[::(len(x_unique) // num_points)]
    y_max = np.array([np.max(y_pts[x_pts == x]) for x in x_unique])
    y_min = np.array([np.min(y_pts[x_pts == x]) for x in x_unique])

    if how == "center":
        samples = np.stack([x_unique, (y_max + y_min) / 2], axis=-1)
    elif how == "boundary":
        samples_upper = np.stack([x_unique, y_max + deviate], axis=-1)
        samples_lower = np.stack([x_unique, y_min - deviate], axis=-1)
        samples = np.concatenate([samples_upper, samples_lower])
    else:
        return NotImplementedError

    return samples


def prepare_image(image: torch.Tensor, transform: ResizeLongestSide):
    image = transform.apply_image(image.permute(1, 2, 0).cpu().numpy())
    image = torch.as_tensor(image)  # (H, W, C)
    return image.permute(2, 0, 1).contiguous()


def save_batch():
    pass


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--raster-root", type=str, required=True)
    parser.add_argument("--vector-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    sam.to(device=device)

    naip_dataset = NAIP(root=args.raster_root)
    river_dataset = RiverBank(root=args.vector_root, res=naip_dataset.res, crs=naip_dataset.crs, )

    dataset = naip_dataset & river_dataset
    print(len(naip_dataset), len(river_dataset), len(dataset))
    sampler = GridGeoSampler(dataset, size=256, stride=256)
    dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples, batch_size=args.batch_size)
    print(len(dataloader))

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # padding to prevent off-by-one errors
        if batch["mask"].shape[1:] != batch["image"].shape[2:]:
            # (B, H, W) v.s. (B, C, H, W)
            batch["mask"] = F.pad(
                batch["mask"],
                pad=(
                    0, batch["image"].size(2) - batch["mask"].size(1),
                    0, batch["image"].size(3) - batch["mask"].size(2)
                )
            )

        b, c, h, w = batch["image"].shape
        if c == 4:
            batch["image"] = batch["image"][:, :3, :, :]

        # Remove blank patches
        for j in range(b):
            fig = naip_dataset.plot(
                {"image": batch["image"][j], "mask": batch["mask"][j]}, suptitle=batch["image"][j].shape
            )
            fig.savefig(os.path.join(args.out_dir, f"original_{str(i).zfill(5)}_{str(j).zfill(2)}.png"))
            plt.close(fig)

        raster_masks = torch.count_nonzero(batch["image"], dim=(1, 2, 3)) / batch["image"][0].numel() > 0.3
        vector_masks = torch.any(torch.flatten(batch["mask"], start_dim=1), dim=1)
        data_masks = torch.logical_and(raster_masks, vector_masks)

        batch["bbox"] = np.array(batch["bbox"])[data_masks]
        batch["image"] = batch["image"][data_masks]
        batch["mask"] = batch["mask"][data_masks]
        batch["original_size"] = [(h, w)] * len(batch["image"])

        if len(batch["image"]) == 0:
            continue

        batch["point_coords"] = mask2points(batch["mask"])
        batch["point_labels"] = [torch.ones(1, x.size(1)) for x in batch["point_coords"]]

        batched_input = unbind_samples({
            "point_coords": [
                resize_transform.apply_coords_torch(x, original_size=(h, w)).to(device) for x in batch["point_coords"]
            ],
            "point_labels": [x.to(device) for x in batch["point_labels"]],
            "image": [prepare_image(x, resize_transform).to(device) for x in batch["image"]],
            "original_size": batch["original_size"]
        })

        batched_output = sam(batched_input, multimask_output=False)

        for j, x in enumerate(batched_input):
            fig = naip_dataset.plot({"image": x["image"], "point_coords": x["point_coords"]})
            fig.savefig(os.path.join(args.out_dir, f"transformed_{str(i).zfill(5)}_{str(j).zfill(2)}.png"))
            plt.close(fig)

        minx, miny, maxx, maxy = unary_union([box(x.minx, x.miny, x.maxx, x.maxy) for x in batch["bbox"]]).bounds
        transform = from_origin(
            west=minx, north=maxy, xsize=naip_dataset.res, ysize=naip_dataset.res
        )
        union_window = from_bounds(left=minx, bottom=miny, right=maxx, top=maxy, transform=transform)

        meta = {
            "crs": batch["crs"][0],
            "transform": transform,
            "driver": 'GTiff',
            "dtype": np.uint8,
            "height": union_window.height,
            "width": union_window.width,
            "nodata": 0,
            "count": 3,
            "compress": "lzw"
        }
        j = 0
        with rasterio.open(os.path.join(args.out_dir, f"{str(i).zfill(5)}.tif"), "w", **meta) as f:
            for output, bbox, img, pts in zip(batched_output, batch["bbox"], batch["image"], batch["point_coords"]):
                mask = output["masks"][0, 0]

                window = from_bounds(
                    left=bbox.minx, right=bbox.maxx, bottom=bbox.miny, top=bbox.maxy, transform=transform
                )

                window = window.intersection(union_window)
                f.write(img.detach().cpu().numpy(), window=window)
                f.write_mask(mask.detach().cpu().numpy(), window=window)

                fig = naip_dataset.plot({"image": img, "mask": output["masks"][0, 0]})
                fig.savefig(os.path.join(args.out_dir, f"masked_{str(i).zfill(5)}_{str(j).zfill(2)}.png"))
                plt.close(fig)

                j += 1


if __name__ == "__main__":
    main()
