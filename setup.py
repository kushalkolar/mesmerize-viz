from setuptools import setup, find_packages
from pathlib import Path


install_requires = [
    "fastplotlib[notebook]==0.1.0a16",
    "pygfx==0.1.17",
    "ipydatagrid",
    "tslearn",
]


with open(Path(__file__).parent.joinpath("README.md")) as f:
    readme = f.read()

with open(Path(__file__).parent.joinpath("mesmerize_viz", "VERSION"), "r") as f:
    ver = f.read().split("\n")[0]


classifiers = \
    [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Science/Research"
    ]


setup(
    name='mesmerize-viz',
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    version=ver,
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    author="Kushal Kolar, Caitlin Lewis",
    author_email='',
    url='https://github.com/kushalkolar/mesmerize-viz',
    license='GPL v3.0',
    description='Mesmerize visualization package using fastplotlib'
)
