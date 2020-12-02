# -*- coding: utf-8 -*-

import re

from setuptools import find_packages, setup


with open("exr_info/__init__.py") as f:
    txt = f.read()
    try:
        version = re.findall(r'^__version__ = "([^"]+)"\r?$', txt, re.M)[0]
    except IndexError:
        raise RuntimeError("Unable to determine version.")

setup(
    name="exr_info",
    version=version,
    python_requires=">=3.7.0",
    install_requires=[
        "numpy>=1.18.0",
        "openexr @ git+https://github.com/jamesbowman/openexrpython.git#egg=openexr",
    ],
    description="Helper modules to parse EXR files generated by renders",
    packages=find_packages(include=['exr_info']),
)
