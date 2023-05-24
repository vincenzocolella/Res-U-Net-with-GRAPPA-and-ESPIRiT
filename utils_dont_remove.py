"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""



from pathlib import Path
from typing import Dict

import h5py
import numpy as np


def save_reconstructions(reconstructions: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset("reconstruction", data=recons)


def convert_fnames_to_v2(path: Path):
    """
    Converts filenames to conform to `v2` standard for knee data.

    For a file with name file1000.h5 in `path`, this script simply renames it
    to file1000_v2.h5. This is for submission to the public knee leaderboards.

    Args:
        path: Path with files to be renamed.
    """
    if not path.exists():
        raise ValueError("Path does not exist")

    for fname in path.glob("*.h5"):
        if not fname.name[-6:] == "_v2.h5":
            fname.rename(path / (fname.stem + "_v2.h5"))


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.
    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


