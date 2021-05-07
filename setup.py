# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:23:51 2021

@author: nasta
"""

import os
from setuptools import setup, find_packages

def parse_requirements(file):
    return sorted(set(
        line.partition('#')[0].strip()
        for line in open(os.path.join(os.path.dirname(__file__), file))
    ) - set(''))


def get_version():
    for line in open(os.path.join(os.path.dirname(__file__), 'core', 
                                  '__init__.py')):
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"').strip("'")
    return version


setup(
    name='eotopia-core',
    python_requires='>=3.6',
    version=get_version(),
    description='EOtopia Core',
    author='Nastasja',
    author_email='nastasja.scholz@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    zip_safe=False
)
