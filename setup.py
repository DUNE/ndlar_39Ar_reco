#!/usr/bin/env python

VER = "0.2.1-alpha"

reqs = ["fire", "h5py", "numpy", "scikit-learn", "tqdm", "scipy", "PyYAML"]

import setuptools

setuptools.setup(
    name="ndlar_39Ar_reco",
    version=VER,
    author="Sam Fogarty",
    author_email="samuel.fogarty@colostate.edu",
    description="Reconstruction code for low-energy activity in ND-LAr prototypes",
    url="https://github.com/sam-fogarty/ndlar_39Ar_reco",
    packages=setuptools.find_packages(),
    scripts=["charge_reco/charge_clustering.py"],
    install_requires=reqs,
    classifiers=[
        "Development Status :: 1 - Pre-Alpha",
        "Intended Audience :: by End-User Class :: Developers",
        "Operating System :: Grouping and Descriptive Categories :: OS Independent (Written in an interpreted language)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.6.8',
)
