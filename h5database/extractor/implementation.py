# noinspection PyProtectedMember
from typing import Tuple, Dict, Sequence
import cv2

import numpy as np
from h5database.database.helper import DataExtractor
import PIL
import re
import os
from h5database.skeletal.abstract_extractor import ExtractCallable
import logging

logging.basicConfig(level=logging.DEBUG)
__all__ = ['ExtSuperResolution', 'ExtTissueByMask']


class ExtSuperResolution(ExtractCallable):

    # override
    @staticmethod
    def type_order() -> Sequence[str]:
        return ['img', 'label']

    @staticmethod
    def _label_key_str() -> str:
        return 'label'

    # override
    @staticmethod
    def get_inputs(obj: DataExtractor, file: str, patch_types: Sequence[str], data_shape: Dict[str, Tuple[int, ...]],
                   **kwargs) -> Tuple[object, ...]:
        resize = kwargs['resize']
        interp = kwargs['interp']

        img = cv2.imread(file, cv2.COLOR_BGR2RGB)
        img_down = cv2.resize(img, (0, 0), fx=resize, fy=resize, interpolation=interp)
        img_low_resolution = cv2.resize(img_down, (img.shape[1], img.shape[0]), interpolation=interp)
        return img_low_resolution, img

    # override
    @staticmethod
    def extractor(inputs: Tuple[object, ...], type_order: Sequence[str], obj: DataExtractor, file: str,
                  data_shape: Dict[str, Tuple[int, ...]], **kwargs) \
            -> Tuple[Tuple, np.ndarray, Sequence[str], object]:
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

    @staticmethod
    def get_mask_name(img_full_path: str, mask_dir: str, suffix: str = '_mask', extension: str = 'png'):
        im_base_name = os.path.basename(img_full_path)
        img_name_file_part = os.path.splitext(im_base_name)[0]
        mask_name = f"{img_name_file_part}{suffix}.{extension}"
        full_mask_name = os.path.join(mask_dir, mask_name)
        return full_mask_name

    # override
    @staticmethod
    def type_order() -> Sequence[str]:
        return ['img', 'mask', 'label']

    @staticmethod
    def _label_key_str() -> str:
        return 'label'

    # override
    @staticmethod
    def get_inputs(obj: DataExtractor, file: str, patch_types: Sequence[str], data_shape: Dict[str, Tuple[int, ...]],
                   **kwargs) -> Tuple[np.ndarray, np.ndarray, object]:
        resize = kwargs['resize']
        interp = kwargs.get('interp', PIL.Image.NONE)
        suffix = kwargs.get('mask_suffix', '_mask')
        extension = kwargs.get('mask_ext', 'png')
        mask_dir = kwargs['mask_dir']

        image_whole = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        image_whole = cv2.resize(image_whole, (0, 0), fx=resize, fy=resize,
                                 interpolation=interp)
        mask_name = ExtTissueByMask.get_mask_name(file, mask_dir, suffix=suffix, extension=extension)
        logging.debug(mask_name)
        mask_whole = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask_whole = cv2.threshold(mask_whole, 1, 255, cv2.THRESH_BINARY)[1]
        mask_whole = cv2.resize(mask_whole, (0, 0), fx=resize, fy=resize,
                                interpolation=interp)
        # mask_whole = cv2.cvtColor(mask_whole, cv2.COLOR_BGR2RGB)
        basename = os.path.basename(file)
        assert obj.database.classes is not None, 'Expect non-None classes'
        class_id = \
            [idx for idx in range(len(obj.database.classes)) if
             re.search(obj.database.classes[idx], basename, re.IGNORECASE)][
                0]
        return image_whole, mask_whole, class_id

    # override
    @staticmethod
    def extractor(inputs: Tuple[np.ndarray, Sequence, np.ndarray], type_order: Sequence[str], obj: DataExtractor,
                  file: str, data_shape: Dict[str, Tuple[int, ...]], **kwargs)\
            -> Tuple[Tuple[object, ...], Sequence[bool], Sequence[str], object]:
        # image_whole, class_id, mask_whole = inputs
        stride_size = kwargs['stride_size']
        flatten = kwargs.get('flatten', True)
        dispose = kwargs.get('dispose', True)
        thresh = kwargs.get('tissue_area_thresh', 0)  # high pass
        logging.debug(f'thresh{thresh}')
        patch_out = tuple(
                                                ExtractCallable.extract_patch(groups, data_shape[type_key], stride_size,
                                                                              flatten=flatten)
                                                for groups, type_key in zip(inputs, type_order)
                                                if not np.isscalar(groups)
                                                 )
        patches_im, patches_mask = patch_out
        num_patches = ExtractCallable.patch_numbers(patch_out[0], n_dims=len(data_shape[type_order[0]]))
        labels = np.asarray([inputs[-1]]*num_patches, dtype=np.int)
        # screen by mask
        mask_axis = tuple(
                            range(
                                -1*len(data_shape[type_order[-2]]),
                                0)
                        )
        assert len(mask_axis) > 0
        valid_patch = patches_mask.mean(axis=mask_axis)
        valid_tag = valid_patch >= thresh
        if flatten and dispose:
            logging.debug(valid_tag.shape)
            logging.debug(labels.shape)
            patches_im = patches_im[valid_tag, :]
            patches_mask = patches_mask[valid_tag, :]
            labels = labels[0:valid_tag.sum()]
        else:
            patches_im[valid_tag, :] = 0
        extra_info = None
        assert patches_im.shape[0] == patches_mask.shape[0] and patches_im.shape[0] == len(labels), f"Length mismatch" \
            f"{patches_im.shape[0]}, {patches_mask.shape[0]}, {len(labels)}"
        return (patches_im, patches_mask, labels), valid_tag, type_order, extra_info
