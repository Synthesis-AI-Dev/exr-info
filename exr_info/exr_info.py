"""Data about the various channels in EXR files output by Synthesis' Blender and Vray render engines"""
import enum
import fnmatch
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import Imath
import numpy as np
import OpenEXR


CryptoDef = namedtuple("CryptoDef", ["name", "id"])


class Renderer(enum.Enum):
    BLENDER = 0
    VRAY = 1


class ExrDtype(enum.Enum):
    FLOAT32 = 0
    FLOAT16 = 1


@dataclass
class CryptoLayerMapping:
    R: str
    G: str
    B: str
    A: str


@enum.unique
class ClassIdFace(enum.IntEnum):
    """Mappings of segments of face to IDs in output PNGs.
    These also correspond to the layer name within the EXRs.
    The segment layers are labelled as "segXX.R" within EXR and contain the mask for a single segment (nose, eyes, etc.)
    Eg, mask for cheeks is stored in layer "seg01.R" in EXR.

    Note: These mappings are for v3 of face data, corresponding to July 2020
    """

    BACKGROUND = 0
    CHEEKS = 1
    CHIN = 2
    EARS = 3
    EYES = 4
    EYE_SOCKETS = 5
    FOREHEAD = 6
    HEAD = 7  # This is back of head.
    JAW_UPPER = 8  # Was JAW pre July 2020
    MOUTH = 9
    MOUTH_BAG = 10  # Inside of mouth
    NECK = 11
    NOSE = 12
    NOSTRILS = 13
    SHOULDERS = 14
    SMILE_LINE = 15
    TEMPLES = 16
    UNDERCHIN = 17
    EYELASHES = 18
    JAW_LOWER = 19  # Added in July 2020
    TEETH = 20  # Added in July 2020
    HAIR = 30
    BEARD = 31
    MUSTACHE = 32
    GLASSES = 33
    MASK = 34
    HEADWEAR = 35


@enum.unique
class ClassIdFace(enum.IntEnum):
    """Mappings of segments of face to IDs in output PNGs.
    These also correspond to the layer name within the EXRs.
    The segment layers are labelled as "segXX.R" within EXR and contain the mask for a single segment (nose, eyes, etc.)
    Eg, mask for cheeks is stored in layer "seg01.R" in EXR.

    Note: These mappings are for v3 of face data, corresponding to July 2020
    """

    BACKGROUND = 0  # This is inside of mouth and empty eyes (when background is seen through eyes)
    BROW = 1
    CHEEK_LEFT = 2
    CHEEK_RIGHT = 3
    CHIN = 4
    EAR_LEFT = 5
    EAR_RIGHT = 6
    EYE_LEFT = 7  # This is back of head. Doesn't include face
    EYE_RIGHT = 8
    EYELASHES = 9
    EYELID = 10  # Inside of mouth
    EYES = 11
    FOREHEAD = 12
    HEAD = 13
    JAW = 14
    JOWL = 15
    LIP_LOWER = 16
    LIP_UPPER = 17
    MOUTH = 18
    MOUTHBAG = 19
    NECK = 20
    NOSE = 21
    NOSE_OUTER = 22
    NOSTRILS = 23
    SHOULDERS = 24
    SMILE_LINE = 25
    TEETH = 26
    TEMPLES = 27
    TONGUE = 28
    UNDEREYE = 29

    HAIR = 100
    BEARD = 101
    MUSTACHE = 102
    GLASSES = 103
    MASK = 104
    HEADWEAR = 105


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
        self.color = {"R": None, "G": None, "B": None}
        self.color_denoised_vray = None  # Vray Only
        self.depth = None
        self.face = None
        self.segment_id = None
        # Cryptomatte - Note: 'cryptomatte' channel in EXR (without numerical suffix), is deprecated
        # Note: The number of cryptomatte layers will depend on the "Level" of the cryptomatte (num_layers==ceil(level/2)).
        self.cryptomatte = {"00": None, "01": None, "02": None}
        # Common
        self.normals = {"X": "normals.X", "Y": "normals.Y", "Z": "normals.Z"}

        if renderer == Renderer.BLENDER:
            self.alpha = "alpha.V"  # Don't use "RGBA.A", it is unreliable
            self.color["R"] = "RGBA.R"
            self.color["G"] = "RGBA.G"
            self.color["B"] = "RGBA.B"
            self.depth = "Z.V"
            self.face = "face.V"
            self.segment_id = "segmentindex.V"

        elif renderer == Renderer.VRAY:
            self.alpha = "A"
            # TODO: Change the color channel names if denoising is present?
            self.color["R"] = "R"
            self.color["G"] = "G"
            self.color["B"] = "B"
            self.color_denoised_vray = {"R": "effectsResult.R", "G": "effectsResult.G", "B": "effectsResult.B"}
            self.depth = "Z"
            self.face = "face.R"
            self.segment_id = "segmentindex.R"

            # Cryptomatte can posses more than one cryptomatte.
            # TODO: Multiple cryptomatte definitions can be present in file. Add methods to recognise presence of
            #       multiple definitions and extract a crypto from each. Each def will have it's own manifest, name and
            #       7-char identifier. Eg: cryptomatte/881c23b/conversion, cryptomatte/881c23b/hash,
            #       cryptomatte/881c23b/manifest, cryptomatte/881c23b/name
            cryptomatte_00 = CryptoLayerMapping(
                R="cryptomatte00.R", G="cryptomatte00.G", B="cryptomatte00.B", A="cryptomatte00.A"
            )
            cryptomatte_01 = CryptoLayerMapping(
                R="cryptomatte01.R", G="cryptomatte01.G", B="cryptomatte01.B", A="cryptomatte01.A"
            )
            cryptomatte_02 = CryptoLayerMapping(
                R="cryptomatte02.R", G="cryptomatte02.G", B="cryptomatte02.B", A="cryptomatte02.A"
            )
            self.cryptomatte = {"00": cryptomatte_00, "01": cryptomatte_01, "02": cryptomatte_02}

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
        dw = self.header["dataWindow"]

        self.height = int(dw.max.y - dw.min.y + 1)
        self.width = int(dw.max.x - dw.min.x + 1)
        self.channels = list(self.header["channels"].keys())

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

    def get_cryptomatte_definitions(self) -> List[CryptoDef]:
        """Get the name and ID of all cryptomatte definitions present in EXR

        A render can contain multiple cryptomattes (each is referred to as a cryptomatte definition). A render engine
        can base each cryptomatte definition on a different criteria, such as material or object id.
        Each cryptomatte definition is given a name and a 7-character ID (which is a hash of the name). The channels
        corresponding to each cryptomatte definition within the EXR will append 2 digits to the cryptomatte name.

        Note: The number of cryptomatte channels (per definition) depends on the "level" of the cryptomatte. The level
              denotes the number of objects a cryptomatte can represent. The default level of 6 results in 3 cryptomatte
              layers (00, 01, 02) in the EXR. Each layer is represented by a channel group with RGBA channels.
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
                .
                .
                VRayCryptomatte_Mtl02.R: FLOAT (1, 1)
        """
        CRYPTOMATTE_IDENTIFIER = "cryptomatte/???????/name"
        crypto_entries = fnmatch.filter(list(self.header.keys()), CRYPTOMATTE_IDENTIFIER)

        crypto_defs = []
        for cry_ent in crypto_entries:
            crypto_id = cry_ent.split("/")[1]
            crypto_name = self.header[cry_ent]
            crypto_defs.append(CryptoDef(name=crypto_name, id=crypto_id))

        return crypto_defs

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
        VRAY_IDENTIFIER = "vrayInfo/"
        if VRAY_IDENTIFIER in list(self.header.keys()):
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
        channels_dict = self.header["channels"]
        for _key in channels_dict:
            channels_str += f"  {_key}: {channels_dict[_key].type}\n"

        return channels_str

    def exr_channel_to_numpy(self, channel_name: str, dtype: ExrDtype = ExrDtype.FLOAT32) -> np.ndarray:
        """Extracts a channel in an EXR file into a numpy array

        Args:
            channel_name (str): The name of the channel to be converted to numpy
            dtype (ExrDtype): Whether the data in channel is of float32 or float16 type

        Returns:
            numpy.ndarray: The extracted channel in form of numpy array with dtype as specified in input parameter.
                           Shape: (H, W).
        """
        if dtype == ExrDtype.FLOAT32:
            point_type = Imath.PixelType(Imath.PixelType.FLOAT)
            np_type = np.float32
        else:
            point_type = Imath.PixelType(Imath.PixelType.HALF)
            np_type = np.float16

        channel_arr = np.frombuffer(self.exr_file.channel(channel_name, point_type), dtype=np_type)
        height, width = self.get_imsize()
        channel_arr = channel_arr.reshape((height, width))

        return channel_arr


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

    return img_srgb
