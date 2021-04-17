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
import altair as alt
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
# choosing file to investigate
genres = ['rock', 'sixeight', 'latin', 'jazz']
filename_options = ['{}_{}'.format(genre, num+1) for genre in genres for num in range(30)]
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
single_file_data_df = pd.DataFrame(rows, columns=columns)
single_file_data_df['genre'] = 'Chosen File'



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


columns = ['note_count_high', 'avg_space_high',
           'note_count_med', 'avg_space_med',
           'note_count_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio']

grpd_all_mdata = all_mdata.groupby(['genre'])[columns].mean()
bar_mdata = grpd_all_mdata.reset_index()
# adding the chosen file's meta data
bar_mdata = pd.concat([bar_mdata, single_file_data_df], ignore_index=True)
# changing genre column to categorical type so can use to order graph correctly
bar_mdata['genre'] = bar_mdata['genre'].replace({'rock':'Rock', 'sixeight':'SixEight', 'latin':'Latin', 'jazz':'Jazz'})
bar_mdata['genre'] = pd.Categorical(bar_mdata['genre'], ['Rock', 'SixEight', 'Latin', 'Jazz', 'Chosen File'])
bar_mdata = bar_mdata.sort_values(['genre']).reset_index(drop=True)

# creating the graphs
bargraph_values = {
	'Note Count': {'note_count_low':'Low Vol Count', 'note_count_med':'Med Vol Count', 'note_count_high':'High Vol Count'},
	'Note Count Ratios': {'med_low_note_ratio':'Med Vol Count / Low Vol Count', 'high_low_note_ratio':'High Vol Count / Low Vol Count', 'high_med_note_ratio':'High Vol Count / Med Vol Count'},
	'Space Between Notes': {'avg_space_low':'Space Between Low Vol Notes', 'avg_space_med':'Space Between Med Vol Notes', 'avg_space_high':'Space Between High Vol Notes'},
	'Space Between Notes Ratios': {'med_low_space_ratio':'Med Vol Space / Low Vol Space','high_low_space_ratio':'High Vol Space / Low Vol Space', 'high_med_space_ratio':'High Vol Space / Med Vol Space'}
}
x=bar_mdata['genre']
for graph_title, graph_columns in bargraph_values.items():
	fig = go.Figure()
	for y_col, y_name in graph_columns.items():
		fig.add_trace(go.Bar(x=x, y=bar_mdata[y_col], name=y_name))
	fig.update_layout(barmode='stack',
					  title=dict(text=graph_title,
								 x=.15,
								 y=.85,
					   			 xanchor='left',
								 yanchor='top'
								 ),
					  title_font_size=20,
	 				  margin=dict(t=30, b=30),
					  height=200,
					  legend=dict(orientation="v",
					    		   yanchor='bottom', y=1.02,
								   xanchor='right', x=1)
					   )
	st.plotly_chart(fig, use_container_width=True)

st.stop()


features = meta_data_df.columns.difference(['genre']).tolist()
st.table(meta_data_df.loc[:,meta_data_df.columns.difference(['genre']).tolist()].T)

actual = meta_data_df['genre'][0]
prediction = classifier.predict(meta_data_df[features])[0]
