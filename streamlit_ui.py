import os
import pandas as pd
import numpy as np
from scipy import stats
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

def plot_title_graphs(plot_data_dict):
	plot12_data_df = plot_data_dict['plot12']
	plot3_data_df = plot_data_dict['plot3']
	plot4_data_df = plot_data_dict['plot4']
	plot5_data_df = plot_data_dict['plot5']

	# Plot 1: initial pull
	fig1, ax = plt.subplots()
	fig1.suptitle('Before', fontsize=20)
	ax.plot(plot12_data_df['lr'])
	# Plot 5: plot of each "note" and which volume it got clustered into
	fig5, ax = plt.subplots()
	fig5.suptitle('After', fontsize=20)
	ax.plot(plot5_data_df['lr_abs_mean_mean'],ls='')
	# ax.bar(data_df.loc[data_df['cluster_label']==0].index, data_df.loc[data_df['cluster_label']==0, 'lr_abs_mean_mean'], color='blue')
	ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==0, 'lr_abs_mean_mean'],ls='', marker='.', color='blue')
	ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==1, 'lr_abs_mean_mean'],ls='', marker='.', color='red')
	ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==2, 'lr_abs_mean_mean'],ls='', marker='.', color='green')
	st.subheader('Before/After Audio File Transformation')
	col1, col2 = st.beta_columns(2)
	with col1:
		st.pyplot(fig1, use_container_width=True)
	with col2:
		st.pyplot(fig5, use_container_width=True)
	st.write('')


def plot_meta_data_graphs(plot_data_dict):
	plot12_data_df = plot_data_dict['plot12']
	plot3_data_df = plot_data_dict['plot3']
	plot4_data_df = plot_data_dict['plot4']
	plot5_data_df = plot_data_dict['plot5']

	# Plot 1: initial pull
	fig1, ax = plt.subplots()
	fig1.suptitle('Original', fontsize=20)
	ax.plot(plot12_data_df['lr'])
	# Plot 2: abs value of initial pull
	fig2, ax = plt.subplots()
	fig2.suptitle('Step 1', fontsize=20)
	ax.plot(plot12_data_df['lr_abs'])
	# Plot 3: smoothed out (mean of each 100 rows)
	fig3, ax = plt.subplots()
	fig3.suptitle('Step 2', fontsize=20)
	ax.plot(plot3_data_df['lr_abs_mean'])
	# Plot 4: mean of each "note" (grouping of numbers with no 0s in between)
	fig4, ax = plt.subplots()
	fig4.suptitle('Step 3', fontsize=20)
	ax.plot(plot4_data_df['lr_abs_mean_mean'])
	# Plot 5: plot of each "note" and which volume it got clustered into
	fig5, ax = plt.subplots()
	fig5.suptitle('Step 4', fontsize=20)
	ax.plot(plot5_data_df['lr_abs_mean_mean'],ls='')
	# ax.bar(data_df.loc[data_df['cluster_label']==0].index, data_df.loc[data_df['cluster_label']==0, 'lr_abs_mean_mean'], color='blue')
	ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==0, 'lr_abs_mean_mean'],ls='', marker='.', color='blue')
	ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==1, 'lr_abs_mean_mean'],ls='', marker='.', color='red')
	ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==2, 'lr_abs_mean_mean'],ls='', marker='.', color='green')
	st.subheader('Full Audio Clip: Step-by-Step Transformation')
	col1, col2, col3 = st.beta_columns(3)
	with col1:
		st.pyplot(fig1, use_container_width=True)
		st.pyplot(fig2, use_container_width=True)
	with col2:
		st.write('')
		st.write('')
		st.write('')
		st.write('')
		st.write('')
		st.write('')
		st.pyplot(fig3, use_container_width=True)
	with col3:
		st.pyplot(fig4, use_container_width=True)
		st.pyplot(fig5, use_container_width=True)
	st.write('')


def plot_single_note_graphs(plot_data_dict):
	# defining dfs
	plot12_df = plot_data_dict['plot12']
	plot3_df = plot_data_dict['plot3']
	plot4_df = plot_data_dict['plot4']
	plot5_df = plot_data_dict['plot5']
	# get diff x axis plot values defined
	first_high_vol_note = plot5_df[plot5_df['cluster_label']==2].index[0]
	highest_vol = plot5_df['lr_abs_mean_mean'].max()
	highest_vol = math.ceil(highest_vol/100) * 100
	start = plot3_df[plot3_df['counter']==first_high_vol_note].index.min()
	end = plot3_df[plot3_df['counter']==first_high_vol_note].index.max()
	all_notes_in_graph_view = plot3_df.loc[start-50:end+50, ['l', 'counter']].groupby('counter').agg('count')
	all_notes_in_graph_view = all_notes_in_graph_view[all_notes_in_graph_view['l']>1].index.tolist()
	first_note_in_view = min(all_notes_in_graph_view)
	last_note_in_view = max(all_notes_in_graph_view)
	# creating graphs
	fig1, ax = plt.subplots()
	fig1.suptitle('Original', fontsize=20)
	ax.plot(plot12_df.loc[(start-50)*100:(end+50)*100, 'lr'])
	fig2, ax = plt.subplots()
	fig2.suptitle('Step 1', fontsize=20)
	ax.plot(plot12_df.loc[(start-50)*100:(end+50)*100, 'lr_abs'])
	fig3, ax = plt.subplots()
	fig3.suptitle('Step 2', fontsize=20)
	ax.plot(plot3_df.loc[start-50:end+50, 'lr_abs_mean'])
	fig4, ax = plt.subplots()
	fig4.suptitle('Step 3', fontsize=20)
	ax.plot(plot4_df.loc[first_note_in_view-1:last_note_in_view+1, 'lr_abs_mean_mean'])
	fig5, ax = plt.subplots()
	fig5.suptitle('Step 4', fontsize=20)
	ax.plot(plot5_df.loc[first_note_in_view-1:last_note_in_view+1,'lr_abs_mean_mean'],ls='', marker='.', color='green')
	ax.set_ylim([0,highest_vol])
	st.subheader('Zoomed View on First Note(s): Step-by-Step Transformation')
	col1, col2, col3 = st.beta_columns(3)
	with col1:
		st.pyplot(fig1, use_container_width=True)
		st.pyplot(fig2, use_container_width=True)
	with col2:
		st.write('')
		st.write('')
		st.write('')
		st.write('')
		st.write('')
		st.write('')
		st.pyplot(fig3, use_container_width=True)
	with col3:
		st.pyplot(fig4, use_container_width=True)
		st.pyplot(fig5, use_container_width=True)
	st.write('')


##################### execution ###########################

title_container = st.beta_container()
title_container.title('Classifying the Genre of a Drum Beat With Machine Learning')
intro_blurb = """Drumming, like all musical instruments, has common "stereo types" that change for each genre of music.
Using singing as an example, if you were to turn on a country song, you expect to hear a southern twang in the singer's voice and some long held-out notes. But
if you were to turn on a rap song, you would expect to hear fast and rhythmic singing. These common "stereo types" exist for drum beats as well. Rock beats are loud with a lot of space between notes,
 while Latin grooves are fast and full of finesse.

 My goal with this personal project was to create a Machine Learning model that could intake an audio recording from my Electronic Drumset and classify the genre of the drum beat I played.
 I did my best to record 120 grooves on my Electronic Drumset, capturing the common stereo types for 4 different genres of music:
 Rock, Shuffle/Funk (SixEight groove), Latin, and Jazz. For each recording, I played 4 measures at varying speeds which resulted in about 9-15 seconds of audio.
  My Electronic Drumset records at 44.1 khz which translates into 44.1k rows of data per second of audio (396.9k-661.5k rows per audio file). After transforming each audio file into meta data,
 an XGBoosted Classification model was trained to be able to correctly identify the genre of each Drum Beat at a 90% accuracy rating."""

genre_stereotypes = """The general stereo types/motifs you can expect in each genre are as follows:
- *Rock:* Big strong hits on the snare a bass drum, with consistent and generally large amounts of space between each note. The Hi-Hat fills in the space between the drum hits.
- *Shuffle/Funk:* The snare drum contains loud hits with a notable amount of "soft" hits inbetween. More bass drum hits compared to rock. Lots of Hi-Hat notes that vary from soft to loud.
- *Latin:* The bass drum follows an ever-constant "heart beat" pattern. A "cha-cha" pattern is played around the snare drum and toms. Fast and consistent notes are played on the cymbals. The Hi-Hat sports a constant "chick" (think boom-chick-boom-chick) pattern.
- *Jazz:* There are little to no bass drum hits. Snare drum hits seem to be played at random and are meant as a form of embellishment, varying from soft to loud. A constant shuffle-like pattern is played on the cymbal with the Hi-Hat being played on the backbeat
 (where you would normally expect to hear a snare drum in any rock beat)."""

with title_container.beta_expander('Introduction', expanded=True):
	st.write(intro_blurb)
with title_container.beta_expander('Genre "Stereotypes"', expanded=True):
	st.write(genre_stereotypes)
title_container.subheader('Choose a file on the left side of the screen, then listen to the groove and guess the genre.\nView the results below to see if you and the ML model guessed the same!')
# choosing file to investigate
genres = ['rock', 'sixeight', 'latin', 'jazz']
filename_options = ['{}_{} (used for train/test)'.format(genre, num+1) for genre in genres for num in range(30)]
filename_options = ['drumbeat_{}'.format(num+1) for num in range(8)] + filename_options
wav_filename = st.sidebar.selectbox('Choose a .wav file:', filename_options)
if wav_filename == '':
	st.stop()
wav_filename = wav_filename.replace("'", "").replace(' (used for train/test)', '')
wav_filename = wav_filename.replace('.wav', '') + '.wav'

# display audio
audio_file = open('groove_samples/' + wav_filename, 'rb')
audio_bytes = audio_file.read()
st.sidebar.audio(audio_bytes, format='audio/ogg')

rows = []
row, plot_data_dict = create_meta_data.create_meta_data(wav_filename)
rows.append(row)

columns = ['note_count_high', 'note_hz_high', 'avg_space_high',
           'note_count_med', 'note_hz_med', 'avg_space_med',
           'note_count_low', 'note_hz_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio', 'med_vol_scale',
           'genre']
single_file_data_df = pd.DataFrame(rows, columns=columns)
single_file_data_df['genre'] = wav_filename


st.header('Data Transformation')
plot_title_graphs(plot_data_dict)
plot_meta_data_graphs(plot_data_dict)
plot_single_note_graphs(plot_data_dict)
st.write('')
st.write('')
st.write('')



all_mdata = pd.read_pickle('meta_data_df.pkl')
classifier = load_in_model('groove_classifier.pkl.dat')

importances = classifier.feature_importances_
features = [
           'note_count_high', 'avg_space_high',
           'note_count_med', 'avg_space_med',
           'note_count_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio', 'med_vol_scale']
importances_df = pd.DataFrame({'feature': features, 'importance':importances})

fig = px.bar(importances_df, x='feature', y='importance')
# st.plotly_chart(fig)


st.header('Comparing Meta Data From Each Genre')
columns = ['note_count_high', 'avg_space_high',
           'note_count_med', 'avg_space_med',
           'note_count_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio']
grpd_all_mdata = all_mdata.groupby(['genre'])[columns].mean()
bar_mdata = grpd_all_mdata.reset_index()
# adding the chosen file's meta data
bar_mdata = pd.concat([bar_mdata, single_file_data_df], ignore_index=True, sort=True)
# changing genre column to categorical type so can use to order graph correctly
bar_mdata['genre'] = bar_mdata['genre'].replace({'rock':'Rock', 'sixeight':'SixEight', 'latin':'Latin', 'jazz':'Jazz'})
bar_mdata['genre'] = pd.Categorical(bar_mdata['genre'], ['Rock', 'SixEight', 'Latin', 'Jazz', wav_filename])
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


features =  [
           'note_count_high', 'avg_space_high',
           'note_count_med', 'avg_space_med',
           'note_count_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio',
           'med_vol_scale']
# model expects columns in abc order
features.sort()
actual = single_file_data_df['genre'][0]
prediction = classifier.predict(single_file_data_df[features])[0]
with title_container.beta_expander('View the Results!'):
	st.subheader('Audio File: `{}`'.format(wav_filename))
	st.subheader('The Model Chose: `{}`'.format(prediction.capitalize()))
title_container.write('')
title_container.write('')
title_container.write('')
