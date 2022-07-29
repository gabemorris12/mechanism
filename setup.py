from setuptools import setup, find_packages
import os

THIS_DIR = r'C:\Users\gmbra\Downloads\Python Programs\mechanism'

VERSION = '1.1.0'

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: MIT License',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python :: 3.10'
]

with open(os.path.join(THIS_DIR, 'readme.md'), 'r') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='mechanism',
    version=VERSION,
    author='Gabe Morris',
    author_email='gabemorris1231@gmail.com',
    description='A package that provides a kinematic analysis of mechanisms and cams and custom tooth profile for spur '
                'gears.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/gabemorris12/mechanism',
    license='MIT',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=['matplotlib', 'scipy'],
    keywords=['mechanism', 'kinematic', 'cams', 'linkages', 'analysis', 'animations'],
    include_package_data=True
)

# See this video: https://youtu.be/v4bkJef4W94
# Don't use the config file though. If you do, you have to use the 'src' folder and that makes things work differently.

# python -m build
# twine upload dist/*
