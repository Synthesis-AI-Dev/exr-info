"""Data about the various channels in EXR files output by Synthesis' Blender and Vray render engines"""
import enum
import fnmatch
import json
from collections import namedtuple, OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import Imath
import numpy as np
import OpenEXR

import cv2

CryptoDef = namedtuple("CryptoDef", ["name", "id"])


class Renderer(enum.Enum):
    BLENDER = 0
    VRAY = 1


class ExrDtype(enum.Enum):
    FLOAT32 = 0
    FLOAT16 = 1


pixel_dtype = {
    ExrDtype.FLOAT32: Imath.PixelType(Imath.PixelType.FLOAT),
    ExrDtype.FLOAT16: Imath.PixelType(Imath.PixelType.HALF),
}

numpy_dtype = {
    ExrDtype.FLOAT32: np.float32,
    ExrDtype.FLOAT16: np.float16,
}


class ExrChannels:
    """This class defines constants to identify channels in the header of the EXR files we render.
    Some channels are named differently in different renderers.
    Default values below according to VRay renderer.

    Note: In Blender and Vray, the beauty pass (RGB image) is saved in sRGB colorspace. Image files are normally
          in sRGB. Other passes (depth, normals, etc) are linear RGB colorspace.
    """

    def __init__(self, renderer):
        """Set constants depending on renderer used

        Args:
            renderer (Renderer): Which renderer we're using.
        """
        if not isinstance(renderer, Renderer):
            raise ValueError(f"Input renderer ({renderer}) must be of type enum {Renderer}")

        # Renderer Dependent
        self.alpha = None
        self.rgb = None
        self.rgb_denoised_vray = None  # Vray Only. If denoising present, denoised RGB stored in different channels.
        self.depth = None
        self.normals = None
        self.normals_corrective_factor = None  # Vray only

        if renderer == Renderer.BLENDER:
            self.rgb = ["RGBA.R", "RGBA.G", "RGBA.B"]
            self.alpha = "alpha.V"  # Don't use "RGBA.A", it is unreliable
            self.depth = "Z.V"
            self.normals = ["normals.X", "normals.Y", "normals.Z"]

        elif renderer == Renderer.VRAY:
            self.rgb = ["R", "G", "B"]
            self.rgb_denoised_vray = ["effectsResult.R", "effectsResult.G", "effectsResult.B"]
            self.alpha = "A"
            self.depth = "Z"
            self.normals = ["normals.X", "normals.Y", "normals.Z"]
            # Vray uses left-handed coord system for normals: X-left, Y-up, Z-behind. We use a
            # right-hand system: X-right, Y-up, Z-behind. Correct it by multiplying each normals channel with
            # the corresponding factor.
            self.normals_corrective_factor = [-1.0, 1.0, 1.0]

        else:
            raise ValueError(f"Unknown Renderer: {renderer}")


class ExrInfo:
    def __init__(self, exr_file: OpenEXR.InputFile):
        """
        Get info about an exr image

        Args:
            exr_file: The opened EXR file object
        """
        self.exr_file = exr_file
        self.header = exr_file.header()

        # Byte-strings cause problems when parsing:
        for key, val in self.header.items():
            if isinstance(val, bytes):
                self.header[key] = str(val, 'utf-8')

        dw = self.header["dataWindow"]

        self.height = int(dw.max.y - dw.min.y + 1)
        self.width = int(dw.max.x - dw.min.x + 1)
        self.channels_dict = self.header["channels"]
        self.channels = list(self.channels_dict.keys())
        self.renderer = self.identify_render_engine()

    @classmethod
    def open(cls, filename: str):
        """Initialize ExrInfo from a file"""
        path_exr = Path(filename)
        if not path_exr.is_file():
            raise ValueError(f"Not a file: {path_exr}")
        exr_file = OpenEXR.InputFile(str(path_exr))
        return cls(exr_file)

    def is_cryptomatte_present(self) -> bool:
        """Check whether any cryptomattes are present in the EXR image

        Cryptomattes are an industry standard to store masks of objects. They can capture anti-aliasing, motion-blur,
        etc. An EXR file can have more than one cryptomatte (each can be based on a different feature, such as material
        or object id).

        Returns:
            bool: True if any cryptomatte present in the EXR

        References:
            https://github.com/Psyop/Cryptomatte
        """
        crypto_defs = self.get_cryptomatte_definitions()
        if len(crypto_defs) == 0:
            return False
        return True

    def get_cryptomatte_definitions(self) -> Dict[CryptoDef]:
        r"""Get the name and ID of all cryptomatte definitions present in EXR

        A render can contain multiple cryptomattes (each is referred to as a cryptomatte definition). A render engine
        can base each cryptomatte definition on a different criteria, such as material or object id.

        Each cryptomatte definition is given a name and a 7-character ID (which is a hash of the name). The channels
        corresponding to each cryptomatte definition within the EXR will append 2 digits to the cryptomatte name.

        Note:
            The number of cryptomatte channels (per definition) depends on the "level" of the cryptomatte. The level
            denotes the number of objects a cryptomatte can represent. The default level of 6 results in 3 cryptomatte
            layers (00, 01, 02) in the EXR. Each layer is represented by a group of 4 channels (R, G, B, A).

                ``num_layers = math.ceil(level / 2)``

        Returns:
            list(namedtuple(name, id)): A list of cryptomatte definitions

        Example:
            EXR Header:
            {
                ...
                cryptomatte/881c23b/manifest:  b'{"smile_line_25":"3bfab174", "ear_left_5":"883129ec", ... }'
                cryptomatte/881c23b/name: b'cryptomatte'
                ...
                cryptomatte/88ce693/manifest: b'{"Leather_black":"7e59982c", "Fabric":"63abf250", ... }'
                cryptomatte/88ce693/name: b'VRayCryptomatte_Mtl'
            }

            Channels:
                cryptomatte00.A: FLOAT (1, 1)
                cryptomatte00.B: FLOAT (1, 1)
                cryptomatte00.G: FLOAT (1, 1)
                cryptomatte00.R: FLOAT (1, 1)
                cryptomatte01.A: FLOAT (1, 1)
                .
                .
                cryptomatte02.R: FLOAT (1, 1)

                ...
                VRayCryptomatte_Mtl00.A: FLOAT (1, 1)
                VRayCryptomatte_Mtl00.B: FLOAT (1, 1)
                .
                .
                VRayCryptomatte_Mtl02.R: FLOAT (1, 1)
        """
        CRYPTOMATTE_IDENTIFIER = "cryptomatte/???????/name"
        crypto_entries = fnmatch.filter(list(self.header.keys()), CRYPTOMATTE_IDENTIFIER)

        crypto_defs = {}
        for cry_ent in crypto_entries:
            crypto_id = cry_ent.split("/")[1]
            crypto_name = self.header[cry_ent]
            crypto_defs[crypto_name] = CryptoDef(name=crypto_name, id=crypto_id)

        return crypto_defs

    def get_cryptomatte_manifest(self, crypto_def: CryptoDef) -> OrderedDict:
        """Get the manifest from EXR header for given cryptomatte definition
        The manifest contains all the objects represented within the cryptomatte and its associated unique ID.
        """
        manifest_key = f"cryptomatte/{crypto_def.id}/manifest"
        manifest = json.loads(self.header[manifest_key], object_pairs_hook=OrderedDict)
        return manifest

    def get_cryptomatte_channels(self, crypto_def: CryptoDef) -> List[str]:
        """Extract list of all the cryptomatte channels, sorted by layer and in order RGBA per layer.
        The ordered channels are used to extract the object masks from a cryptomatte.

        Usage:
            >>> self.get_cryptomatte_channels(CryptoDef(name='crypto', id='xxxxxxx'))
            [crypto00.R, crypto00.G, crypto00.B, crypto00.A, crypto01.R, crypto01.G, ..., crypto02.A]
        """
        wildcard = f"{crypto_def.name}??.?"  # Example: cryptomatte00.R
        crypto_channels = fnmatch.filter(self.channels, wildcard)

        # Crypto channels should be in order RGBA
        crypto_channels = sorted(crypto_channels, key=self._channel_sort_key)

        return crypto_channels

    @staticmethod
    def _channel_sort_dictionary(key: str):
        if key == 'R' or key == 'r':
            return "0000000001"
        elif key == 'G' or key == 'g':
            return "0000000002"
        elif key == 'B' or key == 'b':
            return "0000000003"
        elif key == 'A' or key == 'a':
            return "0000000004"
        else:
            return key

    def _channel_sort_key(self, chan_name):
        return [self._channel_sort_dictionary(x) for x in chan_name.split(".")]

    def is_vray_denoise_present(self) -> bool:
        """Check whether the image was created using Vray denoising.

        Denoising in Vray adds a number of additional channels to the EXR, such as effectsResult.RGB, defocusAmount,
        noiseLevel, reflectionFilter.RGB, refractionFilter.RGB, etc.
        The denoised RGB image is stored in the "effectsResult.RGB" channels

        Returns:
            bool: True, if effectsResult.R channel is present in the EXR
        """
        VRAY_DENOISE_IDENTIFIER = "effectsResult.R"
        if VRAY_DENOISE_IDENTIFIER in self.channels:
            return True
        return False

    def identify_render_engine(self) -> Renderer:
        """Identify which render engine was used to render an EXR"""
        VRAY_IDENTIFIER = "vrayInfo/*"
        vray_info = fnmatch.filter(list(self.header.keys()), VRAY_IDENTIFIER)

        if len(vray_info) > 0:
            render_engine = Renderer.VRAY
        else:
            render_engine = Renderer.BLENDER

        return render_engine

    def get_imsize(self) -> Tuple[int, int]:
        """Get the height and width of image within an EXR file

        Returns:
            int, int: Height, Width of image
        """
        return self.height, self.width

    def get_channels_str(self):
        """Get the list of channels as a string for printing"""
        channels_str = "Channels: \n"
        for _key in self.channels_dict:
            channels_str += f"  {_key}: {self.channels_dict[_key].type}\n"

        return channels_str

    def get_header_str(self):
        """Get the list of channels as a string for printing"""
        header_str = "Header: \n"
        for key, val in self.header.items():
            header_str += f"  {key}: {val}\n"

        return header_str

    def get_channel_precision(self, channel_name: str) -> ExrDtype:
        """Get the precision of a channel within the EXR"""
        if channel_name not in self.channels:
            raise TypeError(
                f"There is no channel called '{channel_name}' in EXR file. "
                f"Available channels:\n{self.get_channels_str()}"
            )

        # Use values in dict to find matching key (values are unique)
        for exr_d, pix_d in pixel_dtype.items():
            prec = self.channels_dict[channel_name].type
            if prec == pix_d:
                return exr_d

    def read_channel(self, channel_name: str, cast_dtype: Optional[ExrDtype] = None) -> np.ndarray:
        """Extracts a channel in an EXR file into a numpy array

        Args:
            channel_name (str): The name of the channel to be converted to numpy
            cast_dtype (ExrDtype): If given, cast array to given dtype. Else, dtype inferred from information in header

        Returns:
            numpy.ndarray: The extracted channel in form of numpy array. Shape: (H, W).
        """
        chan_dtype = self.get_channel_precision(channel_name)
        np_type = numpy_dtype[chan_dtype]
        channel_arr = np.frombuffer(self.exr_file.channel(channel_name), dtype=np_type)
        channel_arr = channel_arr.reshape((self.height, self.width))
        channel_arr = channel_arr.copy()  # Arrays read from buffers can be read-only

        if cast_dtype is not None:
            if not isinstance(cast_dtype, ExrDtype):
                raise ValueError(f"Expected type {ExrDtype.__name__}. Got: {type(cast_dtype)}")
            channel_arr = channel_arr.astype(numpy_dtype[cast_dtype])

        return channel_arr

    def read_channels(self, channel_names: Iterator[str], cast_dtype: Optional[ExrDtype] = None) -> List[np.ndarray]:
        """Read a list of channels from EXR into numpy arrays

        Args:
            channel_names: An iterable containing the names of the channels to be read
            cast_dtype (ExrDtype): If given, cast array to given dtype. Else, dtype inferred from information in header

        Returns:
            list(np.ndarray): A list of the extracted channels in numpy, with dtype inferred from information in header.
                              Shape: [H, W]
        """
        # TODO: Can be made more efficient by reading all channels at once: https://excamera.com/articles/26/doc/openexr.html#OpenEXR.InputFile.channels
        channel_dtypes = []
        for channel_name in channel_names:
            channel_dtypes.append(self.get_channel_precision(channel_name))

        channels_list = []
        for chan_name, chan_dtype in zip(channel_names, channel_dtypes):
            np_type = numpy_dtype[chan_dtype]
            channel_arr = np.frombuffer(self.exr_file.channel(chan_name), dtype=np_type)
            channel_arr = channel_arr.reshape((self.height, self.width))

            if cast_dtype is not None:
                if not isinstance(cast_dtype, ExrDtype):
                    raise ValueError(f"Expected type {ExrDtype.__name__}. Got: {type(cast_dtype)}")
                channel_arr = channel_arr.astype(numpy_dtype[cast_dtype])

            channels_list.append(channel_arr.copy())  # Arrays read from buffers can be read-only

        return channels_list


def lin_rgb_to_srgb_colorspace(img_lin_rgb: np.ndarray):
    """Change color space from linear RGB to sRGB colorspace.

    Blender stores data in EXR files in the linear RGB colorspace. However, for display on a monitor
    or storing as PNG/JPG, the display-ready sRGB colorspace is used. This is a helper function for the conversion.

    Args:
        img_lin_rgb: Image in linear RGB color space.
                     Shape: [H, W, 3] or [H, W], range: [0, 1], dtype=(float32, float16)

    Returns:
        numpy.ndarray: Converted image in sRGB color space.

    References:
        Formula for conversion from linear RGb to sRGB colorspace:
            - https://blender.stackexchange.com/questions/65288/convert-openexr-float-to-color-value
            - https://stackoverflow.com/a/12894053/6021054
        Blender Color Management:
            - https://docs.blender.org/manual/en/latest/render/color_management.html
            "File formats such as PNG or JPEG will typically store colors in a color space ready for display (sRGB).
            For intermediate files in production, it is recommended to use OpenEXR files. These are always stored in
            scene linear color spaces, without any data loss. That makes them suitable to store renders that can later
            be composited, color graded and converted to different output formats."
    """
    if not isinstance(img_lin_rgb, np.ndarray):
        raise ValueError("Input img must be a numpy array.")

    valid_dtypes = [np.float32, np.float16]
    if img_lin_rgb.dtype not in valid_dtypes:
        raise ValueError(f"Invalid dtype: {img_lin_rgb.dtype}. dtype of input must be one of: {valid_dtypes}")

    img_srgb = np.where(img_lin_rgb <= 0.0031308, img_lin_rgb * 12.92, 1.055 * np.power(img_lin_rgb, 1 / 2.4) - 0.055)
    img_srgb = np.clip(img_srgb, a_min=0.0, a_max=1.0)

    return img_srgb


def lin_rgb_to_srgb_reinhard_tonemapping(
    img_float_lin: np.ndarray,
    gamma: float = 0.454,
    intensity: float = 2,
    light_adapt: float = 0,
    color_adapt: float = 1,
):
    """Uses Reinhard method for Tone Mapping (i.e. Color Mapping) to convert Linear RGB to sRGB

    Args:
        img_float_lin: RGB image. Shape: [H, W, 3]
        gamma: gamma value for gamma correction. See createTonemap
        intensity: result intensity in [-8, 8] range. Greater intensity produces brighter results.
        light_adapt: light adaptation in [0, 1] range. If 1 adaptation is based only on pixel
            value, if 0 it's global, otherwise it's a weighted mean of this two cases.
        color_adapt: chromatic adaptation in [0, 1] range. If 1 channels are treated independently,
            if 0 adaptation level is the same for each channel.
    """
    tonemapReinhard = cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)
    ldr_img = tonemapReinhard.process(img_float_lin)
    ldr_img = np.clip(ldr_img, a_min=0.0, a_max=1.0)

    return ldr_img
