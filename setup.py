from setuptools import setup, find_packages

VERSION = '0.0.1'
LONG_DESCRIPTION = """
The readme.md file on the original GitHub repository contains better graphics that detail the capabilities of this 
package. This can be found here: https://github.com/gabemorris12/mechanism."""

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: MIT License',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python :: 3.10'
]

setup(
    name='mechanism',
    version=VERSION,
    author='Gabe Morris',
    author_email='gabemorris1231@gmail.com',
    description='A package that provides a kinematic analysis of mechanisms, cams, and gears.',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/gabemorris12/mechanism',
    license='MIT',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=['matplotlib', 'scipy'],
    keywords=['mechanism', 'kinematic', 'cams', 'linkages', 'analysis', 'animations']
)
