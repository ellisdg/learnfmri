import argparse
import os

import ants
from monai.transforms import LoadImage, EnsureChannelFirst, SaveImage

from learnfmri.stats import compute_correlation_coefficient


def transform_image(image_filename, transform_filename, reference_filename, output_filename, interpolation="linear"):
    image = ants.image_read(image_filename)
    reference = ants.image_read(reference_filename)
    transformed_image = ants.apply_transforms(fixed=reference, moving=image, transformlist=[transform_filename],
                                              interpolation=interpolation)
    transformed_image.to_filename(output_filename)
    return output_filename


def compute_fmri_features(fmri_filename, atlas_filename, transform_filename, output_atlas_filename,
                          output_features_filename, z_transform=True, interpolation="linear", overwrite=False):
    if overwrite or not os.path.exists(output_atlas_filename):
        output_atlas_filename = transform_image(atlas_filename, transform_filename, fmri_filename,
                                                output_atlas_filename, interpolation=interpolation)

    if overwrite or not os.path.exists(output_features_filename):
        loader = LoadImage()
        ensure_channel_first = EnsureChannelFirst()
        saver = SaveImage()
        fmri = ensure_channel_first(loader(fmri_filename))
        atlas = ensure_channel_first(loader(output_atlas_filename))
        features = compute_correlation_coefficient(fmri, atlas, z_transform=z_transform)
        saver(features, output_features_filename)
    return output_features_filename


def compute_fmri_features_for_fmri_file(fmri_filename, atlas_filename, atlas_space="MNI152NLin6Asym"):
    """
    1) Finds the transform that maps the atlas to the T1w space
    2) Derives the output atlas filename
    3) Derives the output features filename
    4) Transforms the atlas and computes the features
    :param fmri_filename: should be in the fmriprep derivatives directory in T1w space.
    :param atlas_filename: atlas file
    :param atlas_space: the space of the atlas
    :return:
    """
    subject = os.path.basename(fmri_filename).split("_")[0]
    transform_filename = os.path.abspath(os.path.join(os.path.dirname(fmri_filename), "..", "anat",
                                                      f"{subject}_from-{atlas_space}_to-T1w_mode-image_xfm.h5"))

    if "fmriprep" not in fmri_filename:
        raise ValueError("fmri_filename should be in the fmriprep derivatives directory.")

    output_directory = os.path.dirname(os.path.dirname(fmri_filename)).replace("fmriprep", "learnfmri")
    os.makedirs(os.path.join(output_directory, "anat"), exist_ok=True)
    output_atlas_filename = os.path.join(output_directory, "anat", f"{subject}_space-T1w_atlas.nii.gz")
    os.makedirs(os.path.join(output_directory, "func"), exist_ok=True)
    output_features_filename = os.path.join(output_directory, "func",
                                            os.path.basename(fmri_filename).replace("bold", "features"))
    return compute_fmri_features(fmri_filename, atlas_filename, transform_filename, output_atlas_filename,
                                 output_features_filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the correlation coefficients between the atlas and the fMRI.")
    parser.add_argument("fmriprep_directory", type=str, help="The fmriprep derivatives directory.")
    parser.add_argument("atlas_filename", type=str, help="The atlas filename.")
    return parser.parse_args()


def main():
    import glob
    args = parse_args()
    fmri_files = glob.glob(os.path.join(args.fmriprep_directory, "**", "func",
                                        "*task-rest*_space-T1w_preproc_bold.nii.gz"),
                           recursive=True)
    for fmri_file in fmri_files:
        print(f"Computing features for {fmri_file}")
        compute_fmri_features_for_fmri_file(fmri_file, args.atlas_filename)


if __name__ == "__main__":
    main()
