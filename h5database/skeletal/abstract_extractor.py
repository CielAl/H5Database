from abc import ABC, abstractmethod
# noinspection PyProtectedMember
from sklearn.feature_extraction.image import extract_patches
from typing import Tuple, Dict, Sequence
from operator import mul
from functools import reduce
import skimage
import types
from skimage.color import rgb2gray
import numpy as np
from h5database.database.helper import DataExtractor


class ExtractCallable(ABC):
    KEY_SHAPE = 'data_shape'
    @staticmethod
    def patch_numbers(patches, n_dims: int = 3) -> int:
        return patches.size // reduce(mul, patches.shape[-n_dims:])

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

        patch_shape = params[cls.KEY_SHAPE]
        params = {key: params[key] for key in params if key is not cls.KEY_SHAPE}
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

    @classmethod
    def label_key(cls) -> str:
        key = cls._label_key_str()
        assert key in cls.type_order(), f'Key of label not defined in type order' \
            f'{key}. Expected from {cls.type_order()}'
        return key

    @staticmethod
    @abstractmethod
    def _label_key_str() -> str:
        ...

    @staticmethod
    @abstractmethod
    def get_inputs(obj: DataExtractor, file: str, patch_types: Sequence[str], data_shape: Dict[str, Tuple[int, ...]],
                   **kwargs) -> Tuple[object, ...]:
        ...
