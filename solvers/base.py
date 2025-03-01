"""
base.py - Base class for all maze solvers.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class MazeSolverBase(ABC):
    """
    Abstract base class that all maze solvers inherit from.
    """
