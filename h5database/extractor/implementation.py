# noinspection PyProtectedMember
from typing import Tuple, Dict, Sequence
import cv2

import numpy as np
from h5database.database.helper import DataExtractor
import PIL
import re
import os
from lazy_property import LazyProperty
from h5database.skeletal.abstract_extractor import ExtractCallable

__all__ = ['ExtSuperResolution', 'ExtTissueByMask']


class ExtSuperResolution(ExtractCallable):

    # override
    @staticmethod
    @LazyProperty
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
            -> Tuple[Tuple[object, ...], Sequence[bool], Sequence[str], object]:
        stride_size = kwargs['stride_size']
        flatten = kwargs.get('flatten', True)
        type_order = ExtSuperResolution.type_order
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
    def get_mask_name_default(img_full_path: str, suffix: str = '_mask', extension: str = 'png'):
        img_name = os.path.splitext(img_full_path)[0]
        full_mask_name = f"{img_name}{suffix}.{extension}"
        return full_mask_name

    # override
    @staticmethod
    @LazyProperty
    def type_order() -> Sequence[str]:
        return ['img', 'label', 'mask']

    @staticmethod
    def _label_key_str() -> str:
        return 'label'

    # override
    @staticmethod
    def get_inputs(obj: DataExtractor, file: str, patch_types: Sequence[str], data_shape: Dict[str, Tuple[int, ...]],
                   **kwargs) -> Tuple[np.ndarray, object, np.ndarray]:
        resize = kwargs['resize']
        interp = kwargs.get('interp', PIL.Image.NONE)
        suffix = kwargs.get('mask_suffix', '_mask')
        extension = kwargs.get('mask_ext', 'png')

        image_whole = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        image_whole = cv2.resize(image_whole, (0, 0), fx=resize, fy=resize,
                                 interpolation=interp)
        mask_name = ExtTissueByMask.get_mask_name_default(file, suffix=suffix, extension=extension)
        mask_whole = cv2.cvtColor(cv2.imread(mask_name), cv2.COLOR_BGR2RGB)
        mask_whole = cv2.resize(mask_whole, (0, 0), fx=resize, fy=resize,
                                interpolation=interp)
        basename = os.path.basename(file)
        class_id = \
            [idx for idx in range(len(obj.database.classes)) if
             re.search(obj.database.classes[idx], basename, re.IGNORECASE)][
                0]
        return image_whole, class_id, mask_whole

    # override
    @staticmethod
    def extractor(inputs: Tuple[np.ndarray, Sequence, np.ndarray], type_order: Sequence[str], obj: DataExtractor,
                  file: str, data_shape: Dict[str, Tuple[int, ...]], **kwargs)\
            -> Tuple[Tuple[object, ...], Sequence[bool], Sequence[str], object]:
        # image_whole, class_id, mask_whole = inputs
        stride_size = kwargs['stride_size']
        flatten = kwargs.get('flatten', True)
        thresh = kwargs.get('tissue_area_thresh', 0)  # high pass
        patches_im, labels, patches_mask = tuple(
                                                ExtractCallable.extract_patch(groups, data_shape[type_key], stride_size,
                                                                              flatten=flatten)
                                                for groups, type_key in zip(inputs, type_order)
                                                 )
        # screen by mask
        mask_axis = tuple(range(-1*len(data_shape[type_order[-1]])))
        valid_patch = patches_mask.mean(axis=mask_axis)
        valid_tag = valid_patch < thresh
        patches_im[valid_tag, :] = 0
        extra_info = None
        return (patches_im, labels, patches_mask), valid_tag, type_order, extra_info
