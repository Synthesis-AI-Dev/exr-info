"""Data about the various channels in EXR files output by Synthesis' Blender and Vray render engines"""
import enum
from dataclasses import dataclass

import numpy as np


@dataclass
class CryptoLayerMapping:
    R: str
    G: str
    B: str
    A: str


class Renderer(enum.Enum):
    BLENDER = 0
    VRAY = 1


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
        # Cryptomatte - Note: 'cryptomatte' (without numerical suffix), is deprecated
        # self.cryptomatte = {'00': {}, '01': {}, '02': {}}  # TODO: Refactor cryptomattes into dict of dicts
        self.cryptomatte_00 = None
        self.cryptomatte_01 = None
        self.cryptomatte_02 = None
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
            self.color['R'] = "R"
            self.color['G'] = "G"
            self.color['B'] = "B"
            self.depth = "Z"
            self.face = "face.R"
            self.segment_id = "segmentindex.R"

            self.cryptomatte_00 = CryptoLayerMapping(R='cryptomatte00.R',
                                                     G='cryptomatte00.G',
                                                     B='cryptomatte00.B',
                                                     A='cryptomatte00.A')

            self.cryptomatte_01 = CryptoLayerMapping(R='cryptomatte01.R',
                                                     G='cryptomatte01.G',
                                                     B='cryptomatte01.B',
                                                     A='cryptomatte01.A')

            self.cryptomatte_02 = CryptoLayerMapping(R='cryptomatte02.R',
                                                     G='cryptomatte02.G',
                                                     B='cryptomatte02.B',
                                                     A='cryptomatte02.A')

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
