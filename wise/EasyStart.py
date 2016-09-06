# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:50:55 2016

@author: Mic
"""

from __future__ import division
import sys as sys
import numpy as np
from numpy import *
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib.pyplot import plot
from time import time, sleep

def Range(Start, Stop, Step):
	return arange(Start, Stop+Step, Step)