from platform import version
from setuptools import setup, find_packages, Extension

setup(
    name='spkmeansmodule',
    version='0.1.0',
    author="Roi and Mike",
    author_email="mika.hurvits22@gmail.com",
    description = "Performing Spectral Clustering",
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    Classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    ext_modules=[
        Extension('spkmeansmodule',['spkmeansmodule.c', 'spkmeans.c'])
        ]
)


