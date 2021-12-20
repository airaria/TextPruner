"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py
To create the package for pypi.
1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.
2. Commit these changes with the message: "Release: VERSION"
3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master
4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
5. Check that everything looks correct by uploading the package to the pypi test server:
   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers
6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.
"""

import shutil
from pathlib import Path
import platform

if platform.python_version() < "3.8":
    numpy_require = "numpy < 1.22"
else:
    numpy_require = "numpy > 1.17"

from setuptools import find_packages, setup

_version=open("src/textpruner/__init__.py").readlines()[0].split()[-1].strip("\"'")

setup(
    name="textpruner",
    version=_version,
    author="Ziqing Yang",
    author_email="ziqingyang@gmail.com",
    description="PyTorch-based model pruning toolkit for pre-trained language models",
    long_description="PyTorch-based model pruning toolkit for pre-trained language models",
    #long_description=open("READMEshort.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformers pruning pytorch",
    #license="",
    url="http://textpruner.hfl-rc.com",
    packages=find_packages('src'),
    package_dir={'':'src'},
    install_requires=[
        "transformers >= 4.0",
        "torch >= 1.7",
        "tqdm",
        "sentencepiece",
        "protobuf",
        numpy_require
    ],
    entry_points = {
       'console_scripts': ['textpruner-cli=textpruner.commands.textpruner_cli:main'],
    },
    python_requires=">=3.7",
    classifiers=[
        #"Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
