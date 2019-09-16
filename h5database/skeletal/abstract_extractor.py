from abc import ABC, abstractmethod
# noinspection PyProtectedMember
from sklearn.feature_extraction.image import extract_patches
from skimage.filters import gaussian, laplace
from typing import Tuple, Dict, Sequence, Callable, Any, Iterable
from operator import mul
from functools import reduce
import types
from skimage.color import rgb2gray
import numpy as np
from h5database.database import DataExtractor
import logging

logging.basicConfig(level=logging.DEBUG)


class ExtractCallable(ABC, Callable):
    """
        Superclass of all Callable that extract data (e.g. patches) from the source files.
        Type order annotated in method type_order.
        Override get_inputs should manually match the order of data types in patch_group
        with the given type_order.
    """
    KEY_SHAPE = 'data_shape'
    @staticmethod
    def patch_numbers(patches: np.ndarray, n_dims: int = 3) -> int:
        """
        Examine the number of patches, if dimensionality is specified.
        Args:
            patches (): Groups of Patch data.
            n_dims (): Dimensionality. Default 3.

        Returns:

        """
        if patches.ndim < 2:  # row vectors
            count = patches.shape[0]
        else:
            patch_size_product = reduce(mul, patches.shape[-n_dims:])
            if patch_size_product == 0:  # empty
                count = 0
            else:
                count = patches.size // patch_size_product
        return count

    @staticmethod
    def validate_shape(patch_groups: Sequence[Any], type_order: Sequence[str], data_shape: Dict[str, Tuple]):
        """
        Validate whether the shape of each element of the arrays in the patch_groups matches the definition in the dict
        of data_shape.
        Args:
            patch_groups (): Sequence of different types of data. i.e. (data_type1_array, data_type2_array, ...)
                                Assumption: leading dimension of data_typex_array is the length of the data array.
            type_order ():  The order of types of data in patch_groups.
            data_shape ():  Dictionary of shape of a single element in the data_array given the type.

        Returns:
        Raises:
            AssertionError if shape does not match
        """
        n_dims = tuple(len(data_shape[x]) for x in type_order)
        logging.debug(f"{tuple(data_shape[type_name] for type_name in type_order)}")
        logging.debug(f"{tuple(patches.shape for patches in patch_groups)}")
        validation_result = tuple(patches.shape[-num_dimension:] == data_shape[type_name]
                                  or (len(data_shape[type_name]) < 1 and patches.ndim == 1)
                                  for (patches, num_dimension, type_name) in zip(patch_groups, n_dims, type_order))
        logging.debug(f"{validation_result}")
        assert np.asarray(validation_result).all(), f"Shape mismatched:" \
            f"{list(patches.shape for patches in patch_groups)}. Expect: {data_shape}"

    @staticmethod
    def extract_patch(image: np.ndarray, patch_shape: Tuple[int, ...], stride: int, flatten: bool = True):
        assert not np.isscalar(image), f"does not support scalar input:{image}"
        if len(patch_shape) == 0:
            patch_shape = 1
        logging.debug(f'image_shape, {(image.shape, patch_shape)}')
        insufficient_size = (x < y for (x, y) in zip(image.shape, patch_shape))
        if any(insufficient_size):
            pad_size = tuple(
                        max((y-x)/2, 0)
                        for (x, y) in zip(image.shape, patch_shape))

            pad_size = tuple((int(np.ceil(x)), int(np.floor(x))) for x in pad_size)
            image = np.pad(image, pad_size, 'wrap')
        patches = extract_patches(image, patch_shape, stride)
        if flatten:
            patches = patches.reshape((-1,) + patch_shape)
        return patches

    @staticmethod
    def get_background_by_contrast(img_gray: np.ndarray, sigma: float = 10, smooth_thresh: float = 0.03):
        """
        A naive background/tissue detection by thresholding the mask of Gaussian-smoothed Laplacian mask.
        Helper function of background_sanitize.
        Args:
            img_gray (): Grayscale image.
            sigma ():   Variance of the Gaussian kernel.
            smooth_thresh ():   Threshold of final background mask (smaller-equal)

        Returns:
            background: background of images.
            mask: binary mask where nonzero elements represent background region.
        """
        img_laplace = np.abs(laplace(img_gray))
        # background region has low response: smaller smooth_thresh --> more strict criteria to spot background
        # sigma --> radius. Larger sigma --> more loose. 10 is slower but more tolerant
        mask = gaussian(img_laplace, sigma=sigma) <= smooth_thresh
        # - pixel 1 is the background part
        background = (mask != 0) * img_gray
        background[mask == 0] = 1  # background[mask_background].mean()
        return background, mask

    @classmethod
    def background_sanitize(cls, image: np.ndarray, sigma: float = 10, smooth_thresh: float = 0.03):
        """
        Remove the background pixels for tissue slides (set to 0).
        Args:
            image ():   original image. Either gray or RGB.
            sigma ():   Radius of the Gaussian kernel.
            smooth_thresh (): Threshold of background mask (smaller-equal)

        Returns:
            image: image with background pixels set to 0.
        """
        img_gray = rgb2gray(image)
        background, mask = cls.get_background_by_contrast(img_gray, sigma=sigma, smooth_thresh=smooth_thresh)
        image[mask == 1] = 0
        return image

    @staticmethod
    @abstractmethod
    def extract(inputs: Tuple[object, ...], type_order: Sequence[str], obj: DataExtractor, file: str,
                data_shape: Dict[str, Tuple[int, ...]], **kwargs)\
            -> Tuple[Sequence[Any], Sequence[bool], Sequence[str], object]:
        """
        Extraction using the result of "get_inputs".
        Args:
            inputs ():  Prepared input array obtained from "get_inputs".
            type_order ():  Pre-defined type order of output data.
            obj (): Associated DataExtractor object.
            file ():    Path of Source file.
            data_shape ():  Shape of data per type.
            **kwargs ():

        Returns:

        """
        ...

    @staticmethod
    def validate_type_order(patch_types: Iterable, type_order: Iterable):
        """
        Raise AssertError if the given patch_types and type_order do not match.
        Args:
            patch_types (): Given data-type of obtained patch
            type_order ():  Predefined data-type for the ExtractCallable.

        Returns:

        """
        compare = set(patch_types) - set(type_order)
        assert len(compare) == 0, f'Predefined ordered type is not identical to the patch types ' \
            f'Got:{type_order}. Expect:{patch_types}'

    @classmethod
    def __call__(cls, obj: DataExtractor, file: str) \
            -> Tuple[Sequence[Any], Sequence[bool], Sequence[str], Any]:
        """
        Callable Interface of ExtractCallable.
        Args:
            obj (): DataExtractor object that initiate the call.
            file ():    File path of the source file to extract.

        Returns:
            output (Tuple[Sequence[Any], Sequence[bool], Sequence[str], Any]):
                    Sequence[Any]: a sequence of [data_array_type_1, data_array_type_2, ...].
                    Sequence[bool]: Sequence of validation tag.
                    Any: Any extra information obtained. As a placeholder for future development.
        """
        # Fetch external parameters from field "meta" of
        if isinstance(obj.meta, types.SimpleNamespace):
            params = obj.meta.__dict__
        else:
            params = obj.meta
        patch_types = obj.database.types
        # Define extra parameters.
        patch_shape = params[cls.KEY_SHAPE]
        # Remove the KEY_SHAPE entity from the dict, as not defined in the interface of cls.extract
        params = {key: params[key] for key in params if key is not cls.KEY_SHAPE}
        # Examine the type order.
        type_order = cls.type_order()
        cls.validate_type_order(patch_types, type_order)
        # Prepare the input by the given type order.
        inputs = cls.get_inputs(obj, file, type_order, patch_shape, **params)
        # Extract the ordered output by the ordered input.
        output: Tuple[Sequence[Any], Sequence[bool], Sequence[str], object] = \
            cls.extract(inputs, type_order, obj, file, patch_shape, **params)
        out_data, is_valid, type_order, extra_info = output

        # Examine if the # of types in the output array size matches the # of pre-defined types.
        assert len(out_data) == len(patch_types), f"Number of returned data type mismatched" \
            f"Got:{len(out_data)}vs. Expect:{len(patch_types)}"
        # logging.debug(f"{type_order}")
        # Validate the output shape.
        cls.validate_shape(out_data, type_order, patch_shape)

        # Calculate the size of each data array
        num_patch_group = tuple(cls.patch_numbers(patches, n_dims=len(patch_shape[type_name]))
                                for (patches, type_name) in zip(out_data, type_order))
        # validate that the each patch array has the same number of data points.
        assert (~np.diff(num_patch_group)).all(), f"Number of patches across types mismatched. " \
            f"Got:{num_patch_group}, Types:{type_order}"
        return output

    @staticmethod
    @abstractmethod
    def type_order() -> Sequence[str]:
        """
        Returns:
            Predefined order of the data types.
        """
        ...

    @classmethod
    def label_key(cls) -> str:
        """
            The attribute name of the "label" (output, ground truth, etc.) entity.
            Must be defined in type_order as well.
        Returns:
            key (str): The attribute name of the label.
        Raises:
            AssertionError, if not defined in type_order.
        """
        key = cls._label_key_str()
        assert key in cls.type_order(), f'Key of label not defined in type order' \
            f'{key}. Expected from {cls.type_order()}'
        return key

    @staticmethod
    def _label_key_str() -> str:
        """
        Returns:

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_inputs(obj: DataExtractor, file: str, patch_types: Sequence[str], data_shape: Dict[str, Tuple[int, ...]],
                   **kwargs) -> Tuple[object, ...]:
        """
        Pre-process the input for the extraction. Method "extract" will be performed on the prepared input.
        Args:
            obj (): Associated DataExtractor obj.
            file ():    Path of the source file.
            patch_types (): Data type names of all target inputs.
            data_shape ():  Shapes of data.
            **kwargs ():    Keyword arguments.
        Keyword Args:
            Given by obj.meta. Implemented by derived classes.
        Returns:
            Input patches group.
        """
        ...
