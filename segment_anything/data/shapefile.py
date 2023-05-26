from torchgeo.datasets import VectorDataset


class RiverBank(VectorDataset):
    filename_glob = "*_Stream_Order_Buffer*"
