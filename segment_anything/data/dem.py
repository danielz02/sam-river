from segment_anything.data.naip import RiverBank
from torch.utils.data import Dataset, DataLoader
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler
from torchgeo.datasets import RasterDataset, VectorDataset, stack_samples


class DEMDataset(RasterDataset):
    filename_glob = "_dtm.tif"
    filename_regex = ".*"


class NAIP(RasterDataset):
    filename_glob = "Mackinaw_Main_*"


def main():
    naip_dataset = NAIP(root="/projects/bbkc/danielz/NAIP/Leaf-Off-2011/ROI/2011_Leaf_off/Mackinaw_Main/")
    dem_dataset = RasterDataset(root="/projects/bbkc/danielz/ILHMP/2010s/DTM_ROI/Mackinaw_River_Stream_Order_Buffer/")
    mackinaw_river = RiverBank(root="/projects/bbkc/danielz/river_bank/data/Shapefiles/Mackinaw_2011/")
    intersection_dataset = dem_dataset & mackinaw_river & naip_dataset
    sampler = RandomBatchGeoSampler(intersection_dataset, size=256, batch_size=16)
    dataloader = DataLoader(intersection_dataset, batch_sampler=sampler, collate_fn=stack_samples)

    for batch in dataloader:
        print(batch["image"].shape)
        print(batch["mask"].shape)


if __name__ == "__main__":
    main()
