# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:36:18 2020

@author: danny
"""

import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns
import xgboost as xgb
import graphviz
import math
import pickle
import re
from create_meta_data import *

############################## execution ##############################


classifier = pickle.load(open('groove_classifier.pkl.dat', 'rb'))


wav_filename = input('Enter wav filename:')
wav_filename = wav_filename.replace("'", "")

rows = []
rows.append(create_meta_data(wav_filename))

columns = ['note_count_high', 'note_hz_high', 'avg_space_high',
           'note_count_med', 'note_hz_med', 'avg_space_med',
           'note_count_low', 'note_hz_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio',
           'genre']
meta_data_df = pd.DataFrame(rows, columns=columns)
features = meta_data_df.columns.difference(['genre']).tolist()

actual = meta_data_df['genre'][0]
prediction = classifier.predict(meta_data_df[features])[0]

print('Actual: {}\nPrediction: {}'.format(actual, prediction))
