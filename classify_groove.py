# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:36:18 2020

@author: danny
"""

import pandas as pd
import pickle
from create_meta_data import *
import create_classifier
import code
import streamlit as st
import plotly.express as px
############################## execution ##############################

meta_data_df = pd.read_pickle('meta_data_df.pkl')
classifier = pickle.load(open('groove_classifier.pkl.dat', 'rb'))

# columns = ['note_count_high', 'note_hz_high', 'avg_space_high',
#            'note_count_med', 'note_hz_med', 'avg_space_med',
#            'note_count_low', 'note_hz_low', 'avg_space_low',
#            'high_med_note_ratio', 'high_med_space_ratio',
#            'high_low_note_ratio', 'high_low_space_ratio',
#            'med_low_note_ratio', 'med_low_space_ratio']

importances = classifier.feature_importances_
features = [
           'note_count_high', 'avg_space_high',
           'note_count_med', 'avg_space_med',
           'note_count_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio', 'med_vol_scale']
importances_df = pd.DataFrame({'feature': features, 'importance':importances})
st.dataframe(importances_df)

fig = px.bar(importances_df, x='feature', y='importance')
st.plotly_chart(fig)
st.stop()


















# wav_filename = input('Enter wav filename:')
# wav_filename = wav_filename.replace("'", "")

# rows = []
# rows.append(create_meta_data(wav_filename))

# columns = ['note_count_high', 'note_hz_high', 'avg_space_high',
#            'note_count_med', 'note_hz_med', 'avg_space_med',
#            'note_count_low', 'note_hz_low', 'avg_space_low',
#            'high_med_note_ratio', 'high_med_space_ratio',
#            'high_low_note_ratio', 'high_low_space_ratio',
#            'med_low_note_ratio', 'med_low_space_ratio',
#            'genre']
# meta_data_df = pd.DataFrame(rows, columns=columns)
# features = meta_data_df.columns.difference(['genre']).tolist()

# actual = meta_data_df['genre'][0]
# prediction = classifier.predict(meta_data_df[features])[0]

# print('The drum groove was categorized as: {}'.format(actual, prediction))
