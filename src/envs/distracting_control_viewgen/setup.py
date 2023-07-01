import os
import sys
from setuptools import setup, find_packages

print("Installing distracting_control_viewgen.")

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(
    name='distracting_control_viewgen',
    version='1.0.0',
    packages=find_packages(),
    description='benchmark for view generalization',
    author='sizheyang',
)