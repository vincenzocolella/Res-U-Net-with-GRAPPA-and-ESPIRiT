

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

# +
import fastmri
import numpy as np
import torch
import random
import h5py
from fastmri.data.transforms import center_crop_to_smallest,to_tensor,tensor_to_complex_np
import os


#import bart
# -

from fastmri.data.subsample import MaskFunc


def to_tensor(data: np.ndarray) -> torch.Tensor:
    
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


"""def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    
    return torch.view_as_complex(data).numpy()"""

from pygrappa import grappa
def apply_grappa(masked_kspace, mask):
    """
    Applies GRAPPA algorithm
    References
    ----------
    [1] Griswold, Mark A., et al. "Generalized autocalibrating
       partially parallel acquisitions (GRAPPA)." Magnetic
       Resonance in Medicine: An Official Journal of the
       International Society for Magnetic Resonance in Medicine
       47.6 (2002): 1202-1210.
    Args:
        masked_kspace (torch.Tensor): Multi-coil masked input k-space of shape (num_coils, rows, cols, 2)
        mask (torch.Tensor): Applied mask of shape (1, 1, cols, 1)
    Returns:
        preprocessed_masked_kspace (torch.Tensor): Output of GRAPPA algorithm applied on masked_kspace
    """

    def get_low_frequency_lines(mask):
        l = r = mask.shape[-2] // 2
        while mask[..., r, :]:
            r += 1

        while mask[..., l, :]:
            l -= 1

        return l + 1, r

    l, r = get_low_frequency_lines(mask)
    num_low_freqs = r - l
    pad = (mask.shape[-2] - num_low_freqs + 1) // 2
    calib = masked_kspace[:, :, pad:pad + num_low_freqs].clone()
    preprocessed_masked_kspace = grappa(tensor_to_complex_np(masked_kspace), tensor_to_complex_np(calib), kernel_size=(5, 5), coil_axis=0)
    return to_tensor(preprocessed_masked_kspace)


def apply_mask(data, mask_func, seed=None, padding=None):
    """
    Subsample given k-space by multiplying with a mask.
    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.
    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, :padding[0]] = 0
        mask[:, :, padding[1]:] = 0 # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0 # The + 0.0 removes the sign of the zeros
    return masked_data, mask


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class UnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    max_value: float


class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        kspace_torch = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace = apply_mask(kspace_torch, self.mask_func, seed=seed)[0]
        else:
            masked_kspace = kspace_torch

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)
        
        
        
        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target_torch = to_tensor(target)
            target_torch = center_crop(target_torch, crop_size)
            target_torch = normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])

        return UnetSample(
            image=image,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
        )


def standardize_multi_coil_image(kspace, image):
    """
    Applies Multi-channel standardization to multi-coil data using sensitivity maps estimated with ESPIRiT method,
    as described by AIRS Medical team.
    Args:
        kspace (numpy.array): Multi-coil input k-space of shape (num_coils, rows, cols)
    Returns:
        numpy.array: Standardized data of shape (num_coils+1, rows, cols)
    """
    import subprocess
    from sigpy import mri
    #import bart
    kspace_perm = np.moveaxis(kspace, 0, 2)
    kspace_perm = np.expand_dims(kspace_perm, axis=0)
    # Estimate sensitivity maps with ESPIRiT method
    #kspace_perm = " ".join(str(x) for x in kspace_perm)
    #command = 'my_array=({})'.format(kspace_perm)

    # Use subprocess to execute the command in Bash
    #subprocess.run(['bash', '-c', command], check=True)

    #S = bart(1, "ecalib -d0 -m1", kspace_perm)
    #S = bart_espirit(kspace_perm)
    #subprocess.call(['export', 'kspace_perm="' + kspace_perm + '"'], shell=True)
    #bashCommand = "bart ecalib -d0 -m1 $kspace_perm"
    #process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    #output, error = process.communicate()
    
    #bash_command = "echo $kspace_perm"
    #process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    #output, error = process.communicate()
    
    # Define the command as a list of strings
    #command = "bart ecalib -d0 -m1 my_array"

    # Use subprocess to execute the command
    #subprocess.run(command, check=True)

    # Convert the output to a Python string
    #my_variable = output.decode().strip()

    # Print the value of the Python variable
    #print(my_variable)
    
    #S = my_variable
    
    #S = np.moveaxis(S.squeeze(), 2, 0)

    # Uncomment if you want to use sigpy implementation which supports GPU acceleration
    espirit = mri.app.EspiritCalib(kspace, show_pbar=False)
    S = espirit.run()
    #S = S.get()
    
    # use S to combine the multi-channel data using a conjugate sum
    M_comb = np.sum(np.multiply(np.conj(S), image), axis=0)
    #M_comb = np.sum(np.multiply(np.conj(S), image[:, np.newaxis, :, :, :]), axis=0)
    # Uncomment if you want to include residual images
    # compute residual image to consider sensitivity estimation errors or artifacts
    # M_res = image - np.multiply(S, M_comb)
    M_comb = np.expand_dims(M_comb, axis=0)
    # M_out = np.concatenate((M_comb, M_res))
    return M_comb


'''def bart_espirit(
    ks_input,
    shape=None,
    verbose=False,
    filename_ks_tmp="ks.tmp",
    filename_map_tmp="map.tmp",
):
    """Estimate sensitivity maps using BART ESPIRiT.
    ks_input dimensions: [emaps, channels, kz, ky, kx]
    """
    from bart.python.cfl import readcfl, writecfl
    import subprocess
    file_list = ['ks.tmp.cfl', 'ks.tmp.hdr', 'map.tmp.cfl', 'map.tmp.hdr']
    for file in file_list:
        if os.path.exists(file):
            os.remove(file)
            print(f"{file} has been deleted.")
        else:
            print(f"{file} does not exist.")
    
    if shape is not None:
        ks_input = recon.crop(ks_input, [-1, -1, shape[0], shape[1], -1])
    writecfl(filename_ks_tmp, ks_input)
    cmd = "%s ecalib  %s %s" % ('bart', filename_ks_tmp, filename_map_tmp)
    if verbose:
        print("  %s" % cmd)
    subprocess.call(["bash", "-c", cmd])
    #subprocess.check_output(["bash", "-c", cmd])
    sensemap = readcfl(filename_map_tmp)
    
    print("Sentisitivity Map Estimated")
    
    
    return sensemap'''


class UnetDataTransform_Grappa:
    """
    Data Transformer for training U-Net models.
    """
    
    def __init__(
        self,
        which_challenge: str,
        accelerations: int,
        
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
        
                
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.accelerations = accelerations
        
        

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        
        ############### recover data from pickle if file is already preprocessed in cache
        acc = 4 #variabile ridicola creata da me
        # case pickle or case npy
        if acc == 4:
            import pickle
            file_list = os.listdir('multicoil_values/multicoil_train_grappa_espirit_4x')
            if str(fname+'.pickle') in file_list:
                with open('multicoil_values/multicoil_train_grappa_espirit_4x/'+str(fname)+'.pickle', 'rb') as f:
                    ############# da cancellare 
                    x = pickle.load(f)
                    return UnetSample(
                    image=x[0],
                    target=x[1],
                    mean=x[2],
                    std=x[3],
                    fname=x[4],
                    slice_num=x[5],
                    max_value=x[6],)
                
        else:
        
            file_list = os.listdir('multicoil_values/multicoil_train_grappa_espirit_8x')
            if str(fname+'.npy') in file_list:
                # specify the path where the tuple is saved
                path = "multicoil_values/multicoil_train_grappa_espirit_8x/" + str(fname) + '.npy'

                # use np.load() to read the tuple from the file
                my_tuple = np.load(path, allow_pickle=True)

                return UnetSample(
                    image=my_tuple[0],
                    target=my_tuple[1],
                    mean=my_tuple[2],
                    std=my_tuple[3],
                    fname=my_tuple[4],
                    slice_num=my_tuple[5],
                    max_value=my_tuple[6],
                )

                
                
                
        
           
        
        
        ###############
        kspace_torch = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        
        # apply mask
        if self.mask_func:
            
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace_torch, self.mask_func, seed=seed)
        else:
            masked_kspace = kspace_torch
        
        #####################
        
        print("APPLYING GRAPPA")
        masked_kspace = apply_grappa(masked_kspace, mask)
        # APPLYING GRAPPA !!! 
        
        
        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)
        #standardize image !!!!!
        
        image = standardize_multi_coil_image(tensor_to_complex_np(masked_kspace), tensor_to_complex_np(image))
        image = to_tensor(image)
        print("Data Standardization")
        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        
        

        # normalize target
        if target is not None:
            target_torch = to_tensor(target)
            target_torch = center_crop(target_torch, crop_size)
            target_torch = normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])
        
        if accelerations == 4:
            
            import pickle

            # define your tuple
            my_tuple = UnetSample(
                image=image,
                target=target_torch,
                mean=mean,
                std=std,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
            )

            # specify the path where you want to save the tuple
            path = "multicoil_values/multicoil_train_grappa_espirit_4x/" + str(fname) + '.pickle'

            with open(path, "wb") as file:
                # use pickle.dump() to write the tuple to the file
                pickle.dump(my_tuple, file)

            print(f"The tuple has been saved")
            
        else:
        
            if accelerations == 8:
                print("accelerations 8x")
                # define your tuple
                my_tuple = (image,target_torch,mean,std,fname,slice_num,max_value)

                # specify the path where you want to save the tuple
                path = "multicoil_values/multicoil_train_grappa_espirit_8x/" + str(fname) + '.npy'

                # use np.save() to write the tuple to the file
                np.save(path, my_tuple)

                #print(f"The tuple has been saved")


                return UnetSample(
                    image=image,
                    target=target_torch,
                    mean=mean,
                    std=std,
                    fname=fname,
                    slice_num=slice_num,
                    max_value=max_value,
                )


class UnetDataTransform_Validation_Grappa:
    """
    Data Transformer for training U-Net models.
    """
    
    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
                
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        
        

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        ############### recover data from pickle if file is already preprocessed in cache
        import pickle
        file_list = os.listdir('multicoil_values/multicoil_train_grappa_espirit')
        if str(fname+'.pickle') in file_list:
            with open('multicoil_values/multicoil_train_grappa_espirit/'+str(fname)+'.pickle', 'rb') as f:
                x = pickle.load(f)
                #print('pickle')            
                
                return UnetSample(
                image=x[0],
                target=x[1],
                mean=x[2],
                std=x[3],
                fname=x[4],
                slice_num=x[5],
                max_value=x[6],)
                
        
            
        
        
        ###############
        kspace_torch = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        
        # apply mask
        if self.mask_func:
            
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace_torch, self.mask_func, seed=seed)
        else:
            masked_kspace = kspace_torch
        
        #####################
        
        print("APPLYING GRAPPA")
        masked_kspace = apply_grappa(masked_kspace, mask)
        # APPLYING GRAPPA !!! 
        
        
        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)
        #standardize image !!!!!
        
        #image = standardize_multi_coil_image(tensor_to_complex_np(masked_kspace), tensor_to_complex_np(image))
        #image = to_tensor(image)
        #print("Data Standardization")
        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        
        

        # normalize target
        if target is not None:
            target_torch = to_tensor(target)
            target_torch = center_crop(target_torch, crop_size)
            target_torch = normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])
        
            
        

        return UnetSample(
            image=image,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
        )


class UnetDataTransform_Test_Grappa:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        kspace_torch = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace = apply_mask(kspace_torch, self.mask_func, seed=seed)[0]
        else:
            masked_kspace = kspace_torch

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)
        
        
        
        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target_torch = to_tensor(target)
            target_torch = center_crop(target_torch, crop_size)
            target_torch = normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])

        return UnetSample(
            image=image,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
        )




