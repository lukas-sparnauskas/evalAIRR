from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.7'
DESCRIPTION = 'Comparison of real and simulated AIRR datasets'
setup(
    name="evalAIRR",
    version=VERSION,
    author="Lukas Sparnauskas",
    author_email="<lukas.11sp@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    include_package_data=True,
    license='MIT',
    packages=find_packages(),
    keywords=['python', 'airr', 'simulated data'],
    entry_points={'console_scripts': ['evalairr=evalAIRR.cli.main:run']},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "Environment :: Console",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)