import os
import sys
import warnings

# Warning that arise due to using quiver:
warnings.filterwarnings('ignore', 'divide by zero encountered in double_scalars')
warnings.filterwarnings('ignore', 'invalid value encountered in multiply')

# Warning that arises due to the 'move circle' function:
warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')

from .mechanism import Joint, Mechanism, get_joints, get_sum
from .dataframe import Data, read_csv, print_matrix
from .vectors import Vector
from .cams import Cam
from .gears import SpurGear

THIS_DIR = os.path.dirname(__file__)
sys.path.append(THIS_DIR)

__all__ = ['Data', 'read_csv', 'print_matrix', 'Joint', 'Vector', 'Mechanism', 'get_joints', 'get_sum', 'Cam',
           'SpurGear']
