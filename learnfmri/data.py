from monai.transforms import Compose, ResizeD, CropForegroundD, NormalizeIntensityD, ConcatItemsD, EnsureChannelFirstD
from monai.transforms import ResampleToMatchD
from monai.transforms.io.dictionary import LoadImageD
from monai.data import CacheDataset, DataLoader


default_spatial_size = (128, 128, 128)
default_transforms = Compose([LoadImageD(keys=["main_image1", "main_image2", "antagonist_image"]),
                              EnsureChannelFirstD(keys=["main_image1", "main_image2", "antagonist_image"]),
                              CropForegroundD(keys=["antagonist_image"], source_key="antagonist_image"),
                              ResampleToMatchD(keys=["main_image1", "main_image2"], key_dst="antagonist_image"),
                              ResizeD(keys=["main_image1", "main_image2", "antagonist_image"],
                                      spatial_size=default_spatial_size),
                              NormalizeIntensityD(keys=["main_image1", "main_image2", "antagonist_image"]),
                              ConcatItemsD(keys=["antagonist_image", "main_image1"], name="main_image1"),
                              ConcatItemsD(keys=["antagonist_image", "main_image2"], name="main_image2"),
                              ])
