#!/usr/bin/env python

VER = "0.0.0"

reqs = ["numpy", "sklearn", "collections", "json", "time", "fire", "matplotlib", "pickle", "scipy", "h5py"]

import setuptools

setuptools.setup(
    name="LArNDLE",
    version=VER,
    author="Sam Fogarty",
    author_email="samuel.fogarty@colostate.edu",
    description="Reconstruction software for low-energy events in ND-LAr prototypes",
    url="https://github.com/sam-fogarty/LArNDLE",
    packages=setuptools.find_packages(),
    scripts=["reco/reco.py"],
    install_requires=reqs,
    classifiers=[
        "Development Status :: 1 - Pre-Alpha",
        "Intended Audience :: by End-User Class :: Developers",
        "Operating System :: Grouping and Descriptive Categories :: OS Independent (Written in an interpreted language)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.7',
)
