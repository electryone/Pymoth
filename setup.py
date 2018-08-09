#!/usr/bin/env python3

import setuptools

long_description = """PyTrack is designed to give easy access to information from MOTChallenge data set sequences.
The PyTrack object is essentially a Namespace which stores a series of Sequence objects.
Each sequence object contains of a number of Frames, each containing constituent Instance objects.
Instances can return information such as the bounding box, id number, and appearance patch."""


setuptools.setup(
    name="PyTrack",
    version="0.0.3",
    author="Samuel Westlake",
    author_email="s.t.westlake@cranfield.ac.uk",
    description="A package for handling MOTChallenge data sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samuelwestlake/PyTrack",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
