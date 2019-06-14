from abc import ABC, abstractmethod
# noinspection PyProtectedMember
from sklearn.feature_extraction.image import extract_patches
from typing import Tuple, Dict, Sequence
import cv2
from operator import mul
from functools import reduce
import numpy as np
import skimage
from .util import DataExtractor
import types
import PIL
import re
import os
from skimage.color import rgb2gray
from lazy_property import LazyProperty


class ExtractCallable(ABC):

    @staticmethod
    def patch_numbers(patches, n_dims: int = 3):
        return patches.size / reduce(mul, patches.shape[-n_dims:])

    @staticmethod
    def validate_shape(patch_groups, type_order, data_shape):
        n_dims = tuple(len(data_shape[x]) for x in type_order)
        validation_result = tuple(patches.shape[-num_dimension:] == data_shape[type_name]
                                  for (patches, num_dimension, type_name) in zip(patch_groups, n_dims, type_order))
        assert np.asarray(validation_result).all(), f"Shape mismatched:" \
            f"{list(patches.shape for patches in patch_groups)}. Expect: {data_shape}"

    @staticmethod
    def extract_patch(image: np.ndarray, patch_shape: Tuple[int, ...], stride: int, flatten: bool = True):
        if len(patch_shape) == 0:
            patch_shape = 1
        patches = extract_patches(image, patch_shape, stride)
        if flatten:
            patches = patches.reshape((-1,) + patch_shape)
        return patches

    @staticmethod
    def get_background_by_contrast(img_gray: np.ndarray, sigma: float = 10, smooth_thresh: float = 0.03):
        img_laplace = np.abs(skimage.filters.laplace(img_gray))
        # background region has low response: smaller smooth_thresh --> more strict criteria to spot background
        # sigma --> radius. Larger sigma --> more loose. 10 is slower but more tolerant
        mask = skimage.filters.gaussian(img_laplace, sigma=sigma) <= smooth_thresh
        # - pixel 1 is the background part
        background = (mask != 0) * img_gray
        background[mask == 0] = 1  # background[mask_background].mean()
        return background, mask

    @classmethod
    def background_sanitize(cls, image: np.ndarray, sigma: float = 10, smooth_thresh: float = 0.03):
        img_gray = rgb2gray(image)
        background, mask = cls.get_background_by_contrast(img_gray, sigma=sigma, smooth_thresh=smooth_thresh)
        image[mask == 1] = 0
        return image

    @staticmethod
    @abstractmethod
    def extractor(inputs: Tuple[object, ...], type_order: Sequence[str], obj: DataExtractor, file: str,
                  data_shape: Dict[str, Tuple[int, ...]], **kwargs)\
            -> Tuple[Tuple[object, ...], Sequence[bool], Sequence[str], object]:
        ...

    @staticmethod
    def validate_type_order(patch_types, type_order):
        compare = set(patch_types) - set(type_order)
        assert len(compare) == 0, f'Predefined ordered type is not identical to the patch types ' \
            f'Got:{type_order}. Expect:{patch_types}'

    @classmethod
    def __call__(cls, obj: DataExtractor, file: str) \
            -> Tuple[Tuple[object, ...], Sequence[bool], Sequence[str], object]:
        if isinstance(obj.meta, types.SimpleNamespace):
            params = obj.meta.__dict__
        else:
            params = obj.meta
        patch_types = obj.database.types
        patch_shape = params['data_shape']
        type_order = cls.type_order()
        cls.validate_type_order(patch_types, type_order)
        inputs = cls.get_inputs(obj, file, type_order, patch_shape, **params)

        output: Tuple[Tuple[object, ...], Sequence[bool], Sequence[str], object] = \
            cls.extractor(inputs, type_order, obj, file, patch_shape, **params)
        out_data, is_valid, type_order, extra_info = output

        assert len(out_data) == len(patch_types), f"Number of returned data type mismatched" \
            f"Got:{len(out_data)}vs. Expect:{len(patch_types)}"

        cls.validate_shape(out_data, type_order, patch_shape)
        num_patch_group = tuple(cls.patch_numbers(patches, n_dims=len(patch_shape[type_name]))
                                for (patches, type_name) in zip(out_data, type_order))
        assert (~np.diff(num_patch_group)).all(), f"Number of patches across types mismatched. " \
            f"Got:{num_patch_group}, Types:{type_order}"
        return output

    @staticmethod
    @abstractmethod
    def type_order() -> Sequence[str]:
        ...

    @staticmethod
    @abstractmethod
    def get_inputs(obj: DataExtractor, file: str, patch_types: Sequence[str], data_shape: Dict[str, Tuple[int, ...]],
                   **kwargs) -> Tuple[object, ...]:
        ...


class ExtSuperResolution(ExtractCallable):

    # override
    @staticmethod
    @LazyProperty
    def type_order() -> Sequence[str]:
        return ['img', 'label']

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
    def extractor(inputs: Tuple[np.ndarray, ..., np.ndarray], type_order: Sequence[str], obj: DataExtractor, file: str,
                  data_shape: Dict[str, Tuple[int, ...]], **kwargs)\
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
