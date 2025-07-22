from setuptools import find_packages
from distutils.core import setup

setup(
    name='genesis_sampling',
    version='0.1.0',
    author='Yilang Liu',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='yilang.liu@yale.edu',
    description='Genesis environments for Legged Robots',
    install_requires=['genesis-world',
                      'rsl-rl',
                      'matplotlib']
)
