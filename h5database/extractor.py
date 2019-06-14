from abc import ABC, abstractmethod
from sklearn.feature_extraction.image import extract_patches
from typing import Tuple, Dict, Iterable
import cv2
from operator import mul
from functools import reduce
import numpy as np
import skimage
from .util import DataExtractor
import types
import PIL
import re
from .database import Database
import os
from skimage.color import rgb2gray


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
        patches = extract_patches(image, patch_shape, stride)
        if flatten:
            patches = patches.reshape((-1,) + patch_shape)
        return patches

    @staticmethod
    def get_background_by_contrast(img_gray: np.ndarray, sigma: float = 5, smooth_thresh: float = 0.03):
        img_laplace = np.abs(skimage.filters.laplace(img_gray))
        # background region has low response: smaller smooth_thresh --> more strict criteria to spot background
        # sigma --> radius. Larger sigma --> more loose.
        mask = skimage.filters.gaussian(img_laplace, sigma=sigma) <= smooth_thresh
        background = (mask != 0) * img_gray
        background[mask == 0] = 1  # background[mask_background].mean()
        return background, mask

    @staticmethod
    @abstractmethod
    def extractor(obj: DataExtractor, file: str, patch_types: Iterable[str], data_shape: Dict[str, Tuple[int, ...]],
                  **kwargs) -> Tuple[Tuple[object, ...], Iterable[bool], Iterable[str], object]:
        ...

    @classmethod
    def __call__(cls, obj: DataExtractor, file: str) \
            -> Tuple[Tuple[object, ...], Iterable[bool], Iterable[str], object]:
        if isinstance(obj.meta, types.SimpleNamespace):
            params = obj.meta.__dict__
        else:
            params = obj.meta
        patch_types = obj.database.types
        patch_shape = params['data_shape']
        output: Tuple[Tuple[object, ...], Iterable[bool], Iterable[str], object] = \
            cls.extractor(obj, file, patch_types, patch_shape, **params)
        out_data, is_valid, type_order, extra_info = output

        assert len(out_data) == len(patch_types), f"Number of returned data type mismatched" \
            f"Got:{len(out_data)}vs. Expect:{len(patch_types)}"

        cls.validate_shape(out_data, type_order, patch_shape)
        num_patch_group = tuple(cls.patch_numbers(patches, n_dims=len(patch_shape[type_name]))
                                for (patches, type_name) in zip(out_data, type_order))
        assert (~np.diff(num_patch_group)).all(), f"Number of patches across types mismatched. " \
            f"Got:{num_patch_group}, Types:{type_order}"
        return output


class ExtSuperResolution(ExtractCallable):

    # override
    @staticmethod
    def extractor(obj: DataExtractor, file: str, data_shape: Dict[str, Tuple[int, ...]], **kwargs)\
            -> Tuple[Tuple[object, ...], Iterable[bool], Iterable[str], object]:
        resize = kwargs['resize']
        interp = kwargs['interp']
        stride_size = kwargs['stride_size']
        flatten = kwargs.get('flatten', True)

        img = cv2.imread(file, cv2.COLOR_BGR2RGB)
        img_down = cv2.resize(img, (0, 0), fx=resize, fy=resize, interpolation=interp)
        img_low_resolution = cv2.resize(img_down, (img.shape[1], img.shape[0]), interpolation=interp)

        data_order = [img_low_resolution, img]
        type_order = ['img', 'label']
        patches = tuple(ExtractCallable.extract_patch(data_source, data_shape[type_key], stride_size,
                                                      flatten=flatten)
                        for (data_source, type_key) in zip(data_order, type_order)
                        )

        # shape[-3:end] is the size of a single patch, regardless of flatten
        num_patches = ExtractCallable.patch_numbers(patches[0], n_dims=len(data_shape[type_order[0]]))
        is_valid = np.ones(num_patches, dtype=np.bool)
        extra_info = None
        return patches, is_valid, type_order, extra_info


class ExtTissueByMask(ExtractCallable):

    # override
    @staticmethod
    def extractor(obj: DataExtractor, file: str, data_shape: Dict[str, Tuple[int, ...]], **kwargs) \
            -> Tuple[Tuple[object, ...], Iterable[bool], object]:
        ...
