"""Data about the various channels in EXR files output by Synthesis' Blender and Vray render engines"""
import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import OpenEXR

HEADER_CRYPTOMATTE_IDENTIFIER = 'cryptomatte/'
CHANNEL_CRYPTOMATTE_IDENTIFIER = 'cryptomatte00'
HEADER_VRAY_IDENTIFIER = 'vrayInfo/'
HEADER_VRAY_DENOISE_IDENTIFIER = 'effectsResult.R'


class Renderer(enum.Enum):
    BLENDER = 0
    VRAY = 1


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
            raise ValueError(f'Input renderer ({renderer}) must be of type enum {Renderer}')

        # Renderer Dependent
        self.alpha = None
        self.color = {'R': None, 'G': None, 'B': None}
        self.depth = None
        self.face = None
        self.segment_id = None
        # Cryptomatte - Note: 'cryptomatte' channel in EXR (without numerical suffix), is deprecated
        # Note: The number of cryptomatte layers will depend on the "Level" of the cryptomatte (num_layers==ceil(level/2)).
        self.cryptomatte = {'00': None, '01': None, '02': None}
        # Common
        self.normals = {
            'X': "normals.X",
            'Y': "normals.Y",
            'Z': "normals.Z"
        }

        if renderer == Renderer.BLENDER:
            self.alpha = "alpha.V"  # Don't use "RGBA.A", it is unreliable
            self.color['R'] = "RGBA.R"
            self.color['G'] = "RGBA.G"
            self.color['B'] = "RGBA.B"
            self.depth = "Z.V"
            self.face = "face.V"
            self.segment_id = "segmentindex.V"

        elif renderer == Renderer.VRAY:
            self.alpha = "A"
            # TODO: Change the color channel names if denoising is present?
            self.color['R'] = "R"
            self.color['G'] = "G"
            self.color['B'] = "B"
            self.depth = "Z"
            self.face = "face.R"
            self.segment_id = "segmentindex.R"

            cryptomatte_00 = CryptoLayerMapping(R='cryptomatte00.R',
                                                G='cryptomatte00.G',
                                                B='cryptomatte00.B',
                                                A='cryptomatte00.A')
            cryptomatte_01 = CryptoLayerMapping(R='cryptomatte01.R',
                                                G='cryptomatte01.G',
                                                B='cryptomatte01.B',
                                                A='cryptomatte01.A')
            cryptomatte_02 = CryptoLayerMapping(R='cryptomatte02.R',
                                                G='cryptomatte02.G',
                                                B='cryptomatte02.B',
                                                A='cryptomatte02.A')
            self.cryptomatte = {'00': cryptomatte_00, '01': cryptomatte_01, '02': cryptomatte_02}

        else:
            raise ValueError(f'Unknown Renderer: {renderer}')


class ExrInfo:
    def __init__(self, exr_file: OpenEXR.InputFile):
        """Get info about an exr image
        Args:
            exr_file (OpenEXR.InputFile): The opened EXR file object
        """
        self.exr_file = exr_file
        self.header = exr_file.header()

    @classmethod
    def fromfilename(cls, filename: str):
        """Initialize ExrInfo from a file"""
        path_exr = Path(filename)
        if not path_exr.is_file():
            raise ValueError(f'Not a file: {path_exr}')
        exr_file = OpenEXR.InputFile(str(path_exr))
        return cls(exr_file)

    def is_cryptomatte_present(self) -> bool:
        """Check whether cryptomatte is present in the EXR image

        Returns:
            bool: True if cryptomatte layers present in the EXR
        """
        # Get the manifest (mapping of object names to hash ids)
        cryptomatte_present = False
        for key in self.header:
            if HEADER_CRYPTOMATTE_IDENTIFIER in key:
                cryptomatte_present = True

                # Verify the cryptomatte channels are present
                channels_str = 'Channels: \n'
                channels_dict = self.header['channels']
                cryptomatte_channel_present = False
                for _key in channels_dict:
                    channels_str += f'  {_key}: {channels_dict[_key]}\n'
                    if CHANNEL_CRYPTOMATTE_IDENTIFIER in _key:
                        cryptomatte_channel_present = True
                if not cryptomatte_channel_present:
                    raise ValueError(f'Cryptomatte detected, but cryptomatte channels not present in EXR:\n{channels_str}')

                break

        return cryptomatte_present

    def is_vray_denoise_present(self) -> bool:
        """Check whether the image was created using Vray denoising.

        Denoising in Vray adds a number of additional channels to the EXR, such as effectsResult.RGB, defocusAmount,
        noiseLevel, reflectionFilter.RGB, refractionFilter.RGB, etc.
        The denoised RGB image is stored in the "effectsResult.RGB" channels

        Returns:
            bool: True, if effectsResult.R channel is present in the EXR
        """
        vray_denoise_present = False
        for key in self.header['channels']:
            if HEADER_VRAY_DENOISE_IDENTIFIER in key:
                vray_denoise_present = True
                break

        return vray_denoise_present

    def identify_render_engine(self) -> Renderer:
        render_engine = Renderer.BLENDER
        for key in self.header:
            if HEADER_VRAY_IDENTIFIER in key:
                render_engine = Renderer.VRAY
                break

        return render_engine


def lin_rgb_to_srgb_colorspace(img_lin_rgb):
    """Change color space from linear RGB to sRGB colorspace.

    Blender stores data in EXR files in the linear RGB colorspace. However, for display on a monitor
    or storing as PNG/JPG, the display-ready sRGB colorspace is used. This is a helper function for the conversion.

    Args:
        img_lin_rgb (numpy.ndarray): Image in linear RGB color space.
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
            scene linear color spaces, without any data loss. That makes them suitable to store renders that can later be
            composited, color graded and converted to different output formats."
    """
    if not isinstance(img_lin_rgb, np.ndarray):
        raise ValueError('Input img must be a numpy array.')

    valid_dtypes = [np.float32, np.float16]
    if img_lin_rgb.dtype not in valid_dtypes:
        raise ValueError(f'Invalid dtype: {img_lin_rgb.dtype}. dtype of input must be one of: {valid_dtypes}')

    img_srgb = np.where(img_lin_rgb <= 0.0031308,
                        img_lin_rgb * 12.92,
                        1.055 * np.power(img_lin_rgb, 1 / 2.4) - 0.055)

    return img_srgb
