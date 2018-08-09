#!/usr/bin/env python3

import setuptools

long_description = """pymoth (Multi-Object Tracking Handler) is designed to 
make information from MOTChallenge data set sequences easy to load and access in Python.
The pymoth object is a Namespace which stores a Sequence object for each set of labels in the data set.
Each sequence object contains of a number of Frames, each containing constituent Instance objects.
Instances can return information such as the bounding box, id number, and appearance patch."""


setuptools.setup(
    name="pymoth",
    version="0.0.3",
    author="Samuel Westlake",
    author_email="s.t.westlake@cranfield.ac.uk",
    description="A package for handling MOTChallenge data sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samuelwestlake/pymoth",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
