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
import plotly.express as px
import plotly.graph_objects as go
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


all_mdata = pd.read_pickle('meta_data_df.pkl')
classifier = load_in_model('groove_classifier.pkl.dat')

importances = classifier.feature_importances_
features = [
           'note_count_high', 'avg_space_high',
           'note_count_med', 'avg_space_med',
           'note_count_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio']
importances_df = pd.DataFrame({'feature': features, 'importance':importances})
st.dataframe(importances_df)

fig = px.bar(importances_df, x='feature', y='importance')
st.plotly_chart(fig)


genres = ['rock', 'sixeight', 'latin', 'jazz']
filename_options = ['{}_{}'.format(genre, num+1) for genre in genres for num in range(30)]


st.dataframe(all_mdata)
columns = ['note_count_high', 'avg_space_high',
           'note_count_med', 'avg_space_med',
           'note_count_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio']

sns.set_theme(style="white")
corr = all_mdata[columns].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots()
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

st.pyplot(f)



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
note_count_columns = ['note_count_med', 'note_count_high', 'note_count_low']
note_space_columns = ['avg_space_med','avg_space_high', 'avg_space_low']
ratio_count_columns = ['high_med_note_ratio', 'high_low_note_ratio', 'med_low_note_ratio']
ratio_space_columns = ['high_med_space_ratio', 'high_low_space_ratio', 'med_low_space_ratio']

fig_rock1 = go.Figure()
fig_rock1.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['rock', note_count_columns],
      theta=note_count_columns,
      fill='toself',
      name='rock'
))

fig_sixeight1 = go.Figure()
fig_sixeight1.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['sixeight', note_count_columns],
      theta=note_count_columns,
      fill='toself',
      name='sixeight'
))

fig_latin1 = go.Figure()
fig_latin1.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['latin', note_count_columns],
      theta=note_count_columns,
      fill='toself',
      name='latin'
))


fig_jazz1 = go.Figure()
fig_jazz1.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['jazz', note_count_columns],
      theta=note_count_columns,
      fill='toself',
      name='jazz'
))

fig_rock2 = go.Figure()
fig_rock2.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['rock', note_space_columns],
      theta=note_space_columns,
      fill='toself',
      name='rock'
))

fig_sixeight2 = go.Figure()
fig_sixeight2.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['sixeight', note_space_columns],
      theta=note_space_columns,
      fill='toself',
      name='sixeight'
))

fig_latin2 = go.Figure()
fig_latin2.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['latin', note_space_columns],
      theta=note_space_columns,
      fill='toself',
      name='latin'
))


fig_jazz2 = go.Figure()
fig_jazz2.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['jazz', note_space_columns],
      theta=note_space_columns,
      fill='toself',
      name='jazz'
))

fig_rock3 = go.Figure()
fig_rock3.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['rock', ratio_count_columns],
      theta=ratio_count_columns,
      fill='toself',
      name='rock'
))

fig_sixeight3 = go.Figure()
fig_sixeight3.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['sixeight', ratio_count_columns],
      theta=ratio_count_columns,
      fill='toself',
      name='sixeight'
))

fig_latin3 = go.Figure()
fig_latin3.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['latin', ratio_count_columns],
      theta=ratio_count_columns,
      fill='toself',
      name='latin'
))


fig_jazz3 = go.Figure()
fig_jazz3.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['jazz', ratio_count_columns],
      theta=ratio_count_columns,
      fill='toself',
      name='jazz'
))

fig_rock4 = go.Figure()
fig_rock4.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['rock', ratio_space_columns],
      theta=ratio_space_columns,
      fill='toself',
      name='rock'
))

fig_sixeight4 = go.Figure()
fig_sixeight4.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['sixeight', ratio_space_columns],
      theta=ratio_space_columns,
      fill='toself',
      name='sixeight'
))

fig_latin4 = go.Figure()
fig_latin4.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['latin', ratio_space_columns],
      theta=ratio_space_columns,
      fill='toself',
      name='latin'
))


fig_jazz4 = go.Figure()
fig_jazz4.add_trace(go.Scatterpolar(
      r=normalized_grpd_all_mdata.loc['jazz', ratio_space_columns],
      theta=ratio_space_columns,
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
col1, col2, col3, col4 = st.beta_columns(4)
col1.plotly_chart(fig_rock1, use_column_width=True)
col2.plotly_chart(fig_rock2, use_column_width=True)
col3.plotly_chart(fig_rock3, use_column_width=True)
col4.plotly_chart(fig_rock4, use_column_width=True)
col1.plotly_chart(fig_sixeight1, use_column_width=True)
col2.plotly_chart(fig_sixeight2, use_column_width=True)
col3.plotly_chart(fig_sixeight3, use_column_width=True)
col4.plotly_chart(fig_sixeight4, use_column_width=True)
col1.plotly_chart(fig_latin1, use_column_width=True)
col2.plotly_chart(fig_latin2, use_column_width=True)
col3.plotly_chart(fig_latin3, use_column_width=True)
col4.plotly_chart(fig_latin4, use_column_width=True)
col1.plotly_chart(fig_jazz1, use_column_width=True)
col2.plotly_chart(fig_jazz2, use_column_width=True)
col3.plotly_chart(fig_jazz3, use_column_width=True)
col4.plotly_chart(fig_jazz4, use_column_width=True)
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
