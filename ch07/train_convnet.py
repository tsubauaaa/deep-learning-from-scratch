# coding: utf-8
import sys, os
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import loat_mnist
