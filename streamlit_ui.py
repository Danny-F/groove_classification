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
import streamlit as st

import pandas as pd
import pickle
import create_meta_data
import xgboost
import plotly
# os.chdir('C://Users//danny//Documents/data_science/groove_classification')

@st.cache(hash_funcs={xgboost.sklearn.XGBClassifier: id})
def load_in_model(pickle_name):
	classifier = pickle.load(open(pickle_name, 'rb'))
	return classifier

@st.cache()
def create_meta_data_for_all_files(filename_options):
	rows = []
	for filename in filename_options:
		wav_filename = filename + '.wav'
		rows.append(create_meta_data.create_meta_data(wav_filename))

	columns = ['note_count_high', 'note_hz_high', 'avg_space_high',
	           'note_count_med', 'note_hz_med', 'avg_space_med',
	           'note_count_low', 'note_hz_low', 'avg_space_low',
	           'high_med_note_ratio', 'high_med_space_ratio',
	           'high_low_note_ratio', 'high_low_space_ratio',
	           'med_low_note_ratio', 'med_low_space_ratio',
	           'genre']
	meta_data_df = pd.DataFrame(rows, columns=columns)
	return meta_data_df

##################### execution ###########################



classifier = load_in_model('groove_classifier.pkl.dat')

genres = ['rock', 'sixeight', 'latin', 'jazz']
filename_options = ['{}_{}'.format(genre, num+1) for genre in genres for num in range(30)]

all_mdata = create_meta_data_for_all_files(filename_options)
st.dataframe(all_mdata)
columns = ['note_count_high', 'note_hz_high', 'avg_space_high',
           'note_count_med', 'note_hz_med', 'avg_space_med',
           'note_count_low', 'note_hz_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio']


grpd_all_mdata = all_mdata.groupby(['genre'])[columns].mean()
st.dataframe(grpd_all_mdata)

def normalize_column_data(column_data):
	min = np.min(column_data)
	max = np.max(column_data)
	normalized_data = []
	for value in column_data:
		normalized_value = (((value - min) / (max - min)) * 5) + 1
		normalized_data.append(normalized_value)
	return normalized_data

normalized_grpd_all_mdata = pd.DataFrame()
for col in columns:
	normalized_grpd_all_mdata[col] = normalize_column_data(grpd_all_mdata[col])
normalized_grpd_all_mdata.set_index(grpd_all_mdata.index, drop=True, inplace=True)
st.dataframe(normalized_grpd_all_mdata)

import plotly.graph_objects as go

fig_rock = go.Figure()
fig_rock.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['rock', ['note_count_high', 'note_hz_high', 'avg_space_high',
	             'note_count_med', 'note_hz_med', 'avg_space_med',
	             'note_count_low', 'note_hz_low', 'avg_space_low']],
      theta=columns,
      fill='toself',
      name='rock'
))

fig_sixeight = go.Figure()
fig_sixeight.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['sixeight', ['note_count_high', 'note_hz_high', 'avg_space_high',
	             'note_count_med', 'note_hz_med', 'avg_space_med',
	             'note_count_low', 'note_hz_low', 'avg_space_low']],
      theta=columns,
      fill='toself',
      name='sixeight'
))

fig_latin = go.Figure()
fig_latin.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['latin', ['note_count_high', 'note_hz_high', 'avg_space_high',
	             'note_count_med', 'note_hz_med', 'avg_space_med',
	             'note_count_low', 'note_hz_low', 'avg_space_low']],
      theta=columns,
      fill='toself',
      name='latin'
))


fig_jazz = go.Figure()
fig_jazz.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['jazz', ['note_count_high', 'note_hz_high', 'avg_space_high',
	             'note_count_med', 'note_hz_med', 'avg_space_med',
	             'note_count_low', 'note_hz_low', 'avg_space_low']],
      theta=columns,
      fill='toself',
      name='jazz'
))

# fig.update_layout(
#   polar=dict(
#     radialaxis=dict(
#       visible=True,
#       range=[0, 5]
#     )),
#   showlegend=False
# )
col1, col2 = st.beta_columns(2)
col1.plotly_chart(fig_rock, use_container_width=True)
col2.plotly_chart(fig_sixeight, use_container_width=True)
col1.plotly_chart(fig_latin, use_container_width=True)
col2.plotly_chart(fig_jazz, use_container_width=True)
st.stop()

def create_meta_data_for_all_files(filename_options):
	rows = []
	for filename in filename_options:
		wav_filename = filename + '.wav'
	rows.append(create_meta_data.create_meta_data(wav_filename))

	columns = ['note_count_high', 'note_hz_high', 'avg_space_high',
	           'note_count_med', 'note_hz_med', 'avg_space_med',
	           'note_count_low', 'note_hz_low', 'avg_space_low',
	           'high_med_note_ratio', 'high_med_space_ratio',
	           'high_low_note_ratio', 'high_low_space_ratio',
	           'med_low_note_ratio', 'med_low_space_ratio',
	           'genre']
	meta_data_df = pd.DataFrame(rows, columns=columns)
	return meta_data_df

wav_filename = st.sidebar.selectbox('Choose a .wav file:', filename_options)
if wav_filename == '':
	st.stop()
wav_filename = wav_filename.replace("'", "")
wav_filename = wav_filename.replace('.wav', '') + '.wav'

# display audio
audio_file = open('groove_samples/' + wav_filename, 'rb')
audio_bytes = audio_file.read()
st.sidebar.audio(audio_bytes, format='audio/ogg')

rows = []
rows.append(create_meta_data.create_meta_data(wav_filename))

columns = ['note_count_high', 'note_hz_high', 'avg_space_high',
           'note_count_med', 'note_hz_med', 'avg_space_med',
           'note_count_low', 'note_hz_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio',
           'genre']
meta_data_df = pd.DataFrame(rows, columns=columns)



features = meta_data_df.columns.difference(['genre']).tolist()
st.table(meta_data_df.loc[:,meta_data_df.columns.difference(['genre']).tolist()].T)

actual = meta_data_df['genre'][0]
prediction = classifier.predict(meta_data_df[features])[0]
