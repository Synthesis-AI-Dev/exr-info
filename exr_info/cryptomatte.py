import colorsys
import random
import struct
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

from exr_info import ExrInfo


class Crypto:
    def __init__(self, exr_f: ExrInfo):
        """
        Extract the segment maps from the cryptomatte within an EXR.

        Args:
            exr_f (exr_info.ExrInfo): An ExrInfo object.
        """
        if not isinstance(exr_f, ExrInfo):
            raise ValueError(f"Expect exr_f of type {ExrInfo.__name__}. Got: {type(exr_f)}")
        self.exr_f = exr_f
        self.definitions = self.exr_f.get_cryptomatte_definitions()

        # In the manifest, some entries are automatically added by vray, which should be ignored.
        # "default" should contain all the regions which have not been explicitly assigned a value in the cryptomatte.
        self.IGNORE_OBJS_IN_MANIFEST = ["vrayLightDome", "vrayLightMesh", "default"]

    @staticmethod
    def get_coverage_for_rank(float_id: float, cr_combined: np.ndarray, rank: int) -> np.ndarray:
        """
        Get the coverage mask for a given rank from cryptomatte layers

        Args:
            float_id (float32): The ID of the object
            cr_combined (numpy.ndarray): The cryptomatte layers combined into a single array along the channels axis.
                                         By default, there are 3 layers, corresponding to a level of 6.
            rank (int): The rank, or level, of the coverage to be calculated

        Returns:
            numpy.ndarray: Mask for given coverage rank. Dtype: np.float32, Range: [0, 1]
        """
        id_rank = cr_combined[:, :, rank * 2] == float_id
        coverage_rank = cr_combined[:, :, rank * 2 + 1] * id_rank

        return coverage_rank

    @staticmethod
    def _convert_hex_id_to_float_id(hex_id: str) -> float:
        bytes_val = bytes.fromhex(hex_id)
        float_val = struct.unpack(">f", bytes_val)[0]
        return float_val

    def get_mask_for_id(self, obj_hex_id: str, channels_arr: np.ndarray, level: int = 6) -> np.ndarray:
        """
        Extract mask corresponding to a float id from the cryptomatte layers

        Args:
            obj_hex_id (str): The ID of the object (from manifest).
            channels_arr (numpy.ndarray): The cryptomatte layers combined into a single array along the channels axis.
                                         Each layer should be in acsending order with it's channels in RGBA order.
                                         By default, there are 3 layers, corresponding to a level of 6.
            level (int): The Level of the Cryptomatte. Default is 6 for most rendering engines. The level dictates the
                         max num of objects that the crytomatte can represent. The number of cryptomatte layers in EXR
                         will change depending on level.

        Returns:
            numpy.ndarray: Mask from cryptomatte for a given id. Dtype: np.uint8, Range: [0, 255]
        """
        float_id = self._convert_hex_id_to_float_id(obj_hex_id)

        coverage_list = []
        for rank in range(level):
            coverage_rank = self.get_coverage_for_rank(float_id, channels_arr, rank)
            coverage_list.append(coverage_rank)

        coverage = sum(coverage_list)
        coverage = np.clip(coverage, 0.0, 1.0)
        mask = (coverage * 255).astype(np.uint8)
        return mask

    def get_masks_for_all_objs(self, crypto_def_name: str) -> OrderedDict:
        """
        Get an individual mask of every object in the cryptomatte

        Args:
            crypto_def_name: The name of the cryptomatte definition from which to extract masks

        Returns:
            collections.OrderedDict(str, numpy.ndarray): Mapping from the name of each object to it's anti-aliased mask.
                For mask -> Shape: [H, W], dtype: uint8
        """
        crypto_def = self.definitions[crypto_def_name]

        manifest = self.exr_f.get_cryptomatte_manifest(crypto_def)
        # Clean manifest - Some items in the manifest are added automatically by the render engine.
        for item in self.IGNORE_OBJS_IN_MANIFEST:
            if item in manifest:
                del manifest[item]

        crypto_channels = self.exr_f.get_cryptomatte_channels(crypto_def)
        channels_arr = np.stack(self.exr_f.read_channels(crypto_channels), axis=-1)

        # Number of layers depends on level of cryptomatte: ``num_layers = math.ceil(level / 2)``. Default level = 6.
        # Each layer has 4 channels: RGBA
        num_layers = len(crypto_channels) // 4
        level = 2 * num_layers

        # The objects in manifest are sorted alphabetically to maintain some order.
        # Each obj is assigned an unique ID (per image) for the mask
        obj_names = sorted(manifest.keys())
        obj_masks = OrderedDict()
        for obj_name in obj_names:
            obj_hex_id = manifest[obj_name]
            mask = self.get_mask_for_id(obj_hex_id, channels_arr, level)
            obj_masks[obj_name] = mask

        return obj_masks

    def get_combined_mask(self, crypto_def_name: str) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Get a single mask for semantic segmentation representing all the objects within the scene.
        Each object is represented by a unique integer value, starting from 1. 0 is reserved for background.

        Args:
            crypto_def_name: The name of the cryptomatte definition from which to extract masks

        Returns:
            numpy.ndarray: Mask of all objects. Shape: [H, W], dtype: np.uint16.
            dict: Mapping of the object names to mask IDs for this image.
        """
        obj_masks = self.get_masks_for_all_objs(crypto_def_name)

        # Create a map of obj names to ids
        name_to_mask_id_map = OrderedDict()
        name_to_mask_id_map["background"] = 0  # Background is always class 0
        obj_names = obj_masks.keys()
        for idx, obj_name in enumerate(obj_names):
            name_to_mask_id_map[obj_name] = idx + 1

        # Combine all the masks into single mask without anti-aliasing for semantic segmentation
        masks = np.stack(list(obj_masks.values()), axis=0)  # Shape: [N, H, W]
        background_mask = 255 - masks.sum(axis=0)
        masks = np.concatenate((np.expand_dims(background_mask, 0), masks), axis=0)
        mask_combined = masks.argmax(axis=0)
        mask_combined = mask_combined.astype(np.uint16)

        return mask_combined, name_to_mask_id_map

    @staticmethod
    def apply_random_colormap_to_mask(mask_combined: np.ndarray) -> np.ndarray:
        """
        Apply random colors to each segment in the mask, for visualization
        """

        def random_color() -> List:
            hue = random.random()
            sat, val = 0.7, 0.7
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            rgb = []
            for col in [r, g, b]:
                col_np = np.array(col, dtype=np.float32)
                col_np = (np.clip(col_np * 255, 0, 255)).astype(np.uint8)
                col_list = col_np.tolist()
                rgb.append(col_list)
            return rgb

        num_objects = mask_combined.max() + 1
        colors = [[0, 0, 0]] + [random_color() for _ in range(num_objects - 1)]  # Background is fixed color: black
        mask_combined_rgb = np.take(colors, mask_combined, 0)
        return mask_combined_rgb.astype(np.uint8)
