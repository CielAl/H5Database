# noinspection PyProtectedMember
from typing import Tuple, Dict, Sequence, Any
import cv2

import numpy as np
from h5database.database.database import DataExtractor
import PIL
import re
import os
from h5database.skeletal.abstract_extractor import ExtractCallable
import logging

logging.basicConfig(level=logging.NOTSET)
__all__ = ['ExtSuperResolution', 'ExtTissueByMask']


class ExtSuperResolution(ExtractCallable):
    """
        Implementation of ExtractCallable for Super Resolution data, where
        the input is the blurred image, and the label is the ground truth (original image).
    """

    @staticmethod
    def type_order() -> Sequence[str]:
        """
        Returns:
            Annotation of designed data type order.
        """
        return ['img', 'label']

    @staticmethod
    def _label_key_str() -> str:
        return 'label'

    # override
    @staticmethod
    def get_inputs(obj: DataExtractor, file: str, patch_types: Sequence[str], data_shape: Dict[str, Tuple[int, ...]],
                   **kwargs) -> Tuple[object, ...]:
        """
        Prepare the inputs: low resolution image and the ground truth.
        Args:
            obj (): Associated DataExtractor obj.
            file ():    Path of the source file.
            patch_types (): Data type names of all target inputs.
            data_shape ():  Shapes of data.
        Keyword Args:
            resize  (float): Resize factor in interval of (0, 1) for both width and height.
            interp (int): Interpolation Flags for cv2.resize.
        Returns:

        """
        resize: float = kwargs['resize']
        interp: int = kwargs['interp']

        img = cv2.imread(file, cv2.COLOR_BGR2RGB)
        img_down = cv2.resize(img, (0, 0), fx=resize, fy=resize, interpolation=interp)
        img_low_resolution = cv2.resize(img_down, (img.shape[1], img.shape[0]), interpolation=interp)
        return img_low_resolution, img

    # override
    @staticmethod
    def extract(inputs: Tuple[np.ndarray, np.ndarray], type_order: Sequence[str], obj: DataExtractor, file: str,
                data_shape: Dict[str, Tuple[int, ...]], **kwargs) \
            -> Tuple[Tuple, np.ndarray, Sequence[str], Any]:
        """
        Simple patchification of the low resolution image/ground truth obtained from the "get_inputs".
        Args:
            inputs ():  Input tuple.
            type_order ():  Defined type order for the output.
            obj (): Associated DataExtractor.
            file ():
            data_shape ():
            **kwargs ():
        Keyword Args:
            stride_size (int): Stride size of patch extraction. Determine the overlapping ratio.
            flatten (bool): Whether returns a flattened patch array or retain the original shape, given the
                            patch location. Default is True.
        Returns:
            patches (Tuple): Tuple of output arrays grouped by types in a tuple.
            is_valid (np.ndarray):
            type_order (Sequence[str]): An ordered sequence of type names.
            extra_info (Any): Placeholder for anything else to report.
        """
        stride_size = kwargs['stride_size']
        flatten = kwargs.get('flatten', True)
        type_order = ExtSuperResolution.type_order()
        # todo if scalar
        patches = tuple(ExtractCallable.extract_patch(data_source, data_shape[type_key], stride_size,
                                                      flatten=flatten)
                        for (data_source, type_key) in zip(inputs, type_order)
                        )
        # shape[-3:end] is the size of a single patch, regardless of flatten
        num_patches = ExtractCallable.patch_numbers(patches[0], n_dims=len(data_shape[type_order[0]]))
        is_valid = np.ones(num_patches, dtype=np.bool)
        extra_info = None
        return patches, is_valid, type_order, extra_info


class ExtTissueByMask(ExtractCallable):
    """
        Extraction of Tissue patches from PNG ROI, screened by a corresponding tissue binary mask.
        todo: Refactor and generalize it for data without mask
    """

    # override
    @staticmethod
    def type_order() -> Sequence[str]:
        """
        Returns:
            Annotation of designed data type order.
        """
        return ['img', 'mask', 'label', 'row', 'col']

    @staticmethod
    def mask_key() -> str:
        return 'mask'

    @staticmethod
    def img_key() -> str:
        return 'img'

    @staticmethod
    def _label_key_str() -> str:
        return 'label'

    # override
    @staticmethod
    def get_inputs(obj: DataExtractor, file: str, patch_types: Sequence[str], data_shape: Dict[str, Tuple[int, ...]],
                   **kwargs) -> Tuple[np.ndarray, np.ndarray, object]:
        """
        Prepare the reading of image, mask files and the class id.
        Args:
            obj (): Associated DataExtractor obj.
            file ():    Path of the source file.
            patch_types (): Data type names of all target inputs.
            data_shape ():  Shapes of data.
        Keyword Args:
            resize  (float): Resize factor in interval of (0, 1) for both width and height.
            interp (int): Interpolation Flags for cv2.resize.
            mask_suffix (str): file suffix of masks for the mapping from source_img_name --> mask_name.
                                Default: "_mask"
            mask_ext (str): File extension of mask files. Default: "png".
            mask_dir (str): File Directory of all masks.
        Returns:

        """
        resize = kwargs['resize']
        # noinspection PyUnresolvedReferences
        interp = kwargs.get('interp', PIL.Image.NONE)
        suffix = kwargs.get('mask_suffix', '_mask')
        extension = kwargs.get('mask_ext', 'png')
        mask_dir = kwargs['mask_dir']

        # read and resize source image.
        image_whole = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        image_whole = cv2.resize(image_whole, (0, 0), fx=resize, fy=resize,
                                 interpolation=interp)
        # read and resize the corresponding masks
        mask_name = ExtTissueByMask.get_mask_name(file, mask_dir, suffix=suffix, extension=extension)
        # logging.debug(mask_name)
        mask_whole = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask_whole = cv2.threshold(mask_whole, 1, 255, cv2.THRESH_BINARY)[1]
        mask_whole = cv2.resize(mask_whole, (0, 0), fx=resize, fy=resize,
                                interpolation=interp)
        # mask_whole = cv2.cvtColor(mask_whole, cv2.COLOR_BGR2RGB)
        # basename of the file

        # obtain the class id from the file name.
        basename = os.path.basename(file)
        assert obj.database.classes is not None, 'Expect non-None classes'
        class_id = \
            [idx for idx in range(len(obj.database.classes)) if
             re.search(obj.database.classes[idx], basename, re.IGNORECASE)
             ][0]
        return image_whole, mask_whole, class_id

    @staticmethod
    def valid_coordinate_rc(valid_tag: np.ndarray, dispose_flag: bool) -> Tuple[np.ndarray, np.ndarray]:
        valid_tag = np.atleast_2d(valid_tag)
        if dispose_flag:
            row_ind, col_ind = np.where(valid_tag)
        else:
            row_ind, col_ind = np.where(np.ones_like(valid_tag))
        return row_ind, col_ind

    @staticmethod
    def flatten_helper(patches, patch_shape, flatten_flag: bool):
        if flatten_flag:
            patches = ExtractCallable.flatten_patch(patches, patch_shape)
        return patches

    # override
    @staticmethod
    def extract(inputs: Tuple[np.ndarray, np.ndarray, Any], type_order: Sequence[str], obj: DataExtractor,
                file: str, data_shape: Dict[str, Tuple[int, ...]], **kwargs)\
            -> Tuple[Tuple[object, ...], Sequence[bool], Sequence[str], Any]:
        """
        Args:
            inputs ():  Input tuple.
            type_order ():  Defined type order for the output.
            obj (): Associated DataExtractor.
            file ():
            data_shape ():
            **kwargs ():
        Keyword Args:
            stride_size (int): Stride size of patch extraction. Determine the overlapping ratio.
            flatten (bool): Whether returns a flattened patch array or retain the original shape, given the \
                patch location. Default is True.
            dispose (bool): Flat that whether dispose the "invalid" patches, with valid tag of 0, \
                if tissue% is too low. Explicitly discard the result from the array \
                only if `flatten` is True.
            tissue_area_thresh (bool): Percentage of minimum required tissue area.

        Returns:
            (patches_im, patches_mask, labels): arrays of patchified images/masks/labels.
            valid_tag (np.ndarray): Array of valid tag by each single patch.
            type_order (Sequence[str]): A sequence of ordered type names.
            extra_info (Any): Placeholder for any other extra information to export.
        """
        # image_whole, class_id, mask_whole = inputs
        stride_size = kwargs['stride_size']
        flatten = kwargs.get('flatten', True)
        dispose = kwargs.get('dispose', True)
        thresh = kwargs.get('tissue_area_thresh', 0)  # high pass
        logging.debug(f'thresh{thresh}')

        # extract and order the patches
        patch_out = tuple(
                            ExtractCallable.extract_patch(groups, data_shape[type_key], stride_size,
                                                          flatten=False)
                            for groups, type_key in zip(inputs, type_order)
                            if not np.isscalar(groups)
                             )
        # manually define the order of data in patch_out.
        patches_im, patches_mask = patch_out
        # calculate the # of patch
        num_patches = ExtractCallable.patch_numbers(patch_out[0], n_dims=len(data_shape[type_order[0]]))
        # curate/multiply the label. inputs[-1] is the label, given by the type order.
        labels = np.asarray([inputs[-1]]*num_patches, dtype=np.int)

        # type_order[-1] as the type of mask.
        # mask_axis as the dimension of single masks in the high dimensional mask patch-array.
        # excluded in the mean calculation
        mask_axis = tuple(
                            range(
                                -1*len(data_shape[ExtTissueByMask.mask_key()]),
                                0)
                        )

        assert len(mask_axis) > 0
        # Tissue screening by mask region.
        valid_patch = patches_mask.mean(axis=mask_axis)
        valid_tag = valid_patch >= thresh
        # whether flatten and dispose the output.
        # Assume
        row_ind, col_ind = ExtTissueByMask.valid_coordinate_rc(valid_tag, dispose)
        patches_im = ExtTissueByMask.flatten_helper(patches_im, data_shape[ExtTissueByMask.img_key()], flatten)
        patches_mask = ExtTissueByMask.flatten_helper(patches_mask, data_shape[ExtTissueByMask.mask_key()], flatten)
        if not flatten or not dispose:
            patches_im[valid_tag, :] = 0
        else:
            # todo flatten
            valid_tag = ExtTissueByMask.flatten_helper(valid_tag, (1, 1), True).squeeze()
            patches_im = patches_im[valid_tag, :]
            patches_mask = patches_mask[valid_tag, :]
            labels = labels[0:valid_tag.sum()]

        # validate the length of image patch array and mask patch array
        assert patches_im.shape[0] == patches_mask.shape[0] and patches_im.shape[0] == len(labels), f"Length mismatch" \
            f"{patches_im.shape[0]}, {patches_mask.shape[0]}, {len(labels)}"
        extra_info = None
        return (patches_im, patches_mask, labels, row_ind, col_ind), valid_tag, type_order, extra_info
