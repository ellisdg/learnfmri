import torch


def compute_image_derived_phenotypes(image, atlas):
    atlas = atlas.reshape((atlas.shape[0], -1))
    image = image.reshape((image.shape[0], -1))
    return torch.linalg.pinv(atlas.T) @ image.T


def compute_correlation_coefficient(image, atlas):
    phenotypes = compute_image_derived_phenotypes(image, atlas)[:, None]
