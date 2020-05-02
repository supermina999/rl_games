"""Setup script for rl_games"""

import sys
import os

from setuptools import setup, find_packages

print(find_packages())

setup(name='rl_games',
      packages=[package for package in find_packages()
                if package.startswith('rl_games')],
      version='0.9',
      install_requires=[],
      )


#setup(name='rlgames',
#    version='0.8',
#    description='High Performance Distributed Reinforcement Learning Library',
#    author='',
#    author_email='',
#    packages=find_packages(),
#    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning deep-learning",
#    install_requires = [
#            'gym>=0.10.9',
#            'numpy>=1.15.4',
#            'ray==0.6.6',
#            'tensorboard>=1.14.0',
#            'tensorboardX>=1.6',
#            'opencv-python>=4.1.0.25'
#    ],
#)