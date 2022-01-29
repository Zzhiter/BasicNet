import matplotlib as mpl
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf

from tensorflow import keras

for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")