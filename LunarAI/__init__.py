#Welcome to use LunarAI, a convenient AI tool
#For examples, see LunarAI/examples/
#cmd: python3 -m LunarAI.examples.example_1.py
#Python 3.12 cannot execute this program because of the bug in the interpreter

from LunarAI.ai_libs import *
from LunarAI.ai_libs.array_type import Array
from LunarAI.activations.activations import *
from LunarAI.layers import *
from LunarAI.model import *
from LunarAI.loss.loss import *
from LunarAI.train import *

import LunarAI.ai_libs
import LunarAI.activations.activations
import LunarAI.layers
import LunarAI.model
import LunarAI.examples
import LunarAI.loss
import LunarAI.train

__version__ = '0.0.4'
VERSION = __version__

__all__ = ['ai_libs','activations','layers','model','examples','train']
