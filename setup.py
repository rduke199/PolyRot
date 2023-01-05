import sys
import setuptools
from PolyRot import __version__, __author__, __credits__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
if sys.version_info.major <=3 and sys.version_info.minor <=8:
    pass

setuptools.setup(
    name='PolyRot',
    version=__version__,
    author=__author__,
    author_email='rduke199@gmail.com',
    description='Tools for manipulating and rotating polymers',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/rduke199/PolyRot',
    project_urls={
        "Bug Tracker": "https://github.com/rduke199/PolyRot/issues"
    },
    license=__credits__,
    packages=['PolyRot'],
    install_requires=requirements,
)
