import os
from setuptools import setup

setup(
    name = "complex_tf",
    version = "0.0.1",
    author = "Andy Sarroff",
    author_email = "sarroff@cs.dartmouth.edu",
    description = ("Ops for complex-valued tensorflow."),
    license ="LGPLv3+",
    keywords = "complex neural network",
    url = "",
    packages=['complex_tf'],
    data_files=[('complex_tf/core/kernels',
                 ['complex_tf/core/kernels/complextf.so'])],
    long_description='',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or " \
        "later (LGPLv3+)",
    ],
)
