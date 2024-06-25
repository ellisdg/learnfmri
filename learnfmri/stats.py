import torch


def compute_image_derived_phenotypes(image, atlas):
    """
    Compute the phenotypes derived from the image using the atlas.
    Image should be of shape (n_timepoints, x, y, z).
    Atlas should be of shape (n_atlas_regions, x, y, z).
    Returns tensor of shape (n_atlas_regions, n_timepoints).
    :param image:
    :param atlas:
    :return:
    """
    atlas = atlas.reshape((atlas.shape[0], -1))
    image = image.reshape((image.shape[0], -1))
    return torch.linalg.pinv(atlas.T) @ image.T


def correlation_coefficient(x1, x2):
    """
    Compute the correlation coefficient between two vectors x1 and x2.
    Input tensors must be of shape (channels, n_timepoints).
    Will return a tensor of shape (x1_channels, x2_channels).
    :param x1:
    :param x2:
    :return:
    """
    x1 = x1.unsqueeze(1)
    x2 = x2.unsqueeze(0)
    return torch.cosine_similarity(x1 - x1.mean(dim=2, keepdim=True),
                                   x2 - x2.mean(dim=2, keepdim=True),
                                   dim=2)


def compute_correlation_coefficient(image, atlas, z_transform=False):
    phenotypes = compute_image_derived_phenotypes(image, atlas)

    # reshape the image to be of shape (n_voxels, n_timepoints)
    image = image.reshape((image.shape[0], -1)).T

    # compute correlation coefficients tensor of shape (n_atlas_regions, n_voxels)
    pearson_correlations = correlation_coefficient(phenotypes, image)

    # correlations reshaped to the shape of the atlas
    pearson_correlations = pearson_correlations.reshape(atlas.shape)

    if z_transform:
        # return the fisher-z transformed correlations
        return torch.atanh(pearson_correlations)
    else:
        # return the pearson correlations
        return pearson_correlations
