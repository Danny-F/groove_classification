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
	ax.set_xlabel('Hz')
	ax.set_ylabel('Volume')
	# Plot 5: plot of each "note" and which volume it got clustered into
	fig5, ax = plt.subplots()
	fig5.suptitle('After', fontsize=20)
	ax.plot(plot5_data_df['lr_abs_mean_mean'],ls='')
	ax.set_xlabel('kHz')
	ax.set_ylabel('Volume')
	# ax.bar(data_df.loc[data_df['cluster_label']==0].index, data_df.loc[data_df['cluster_label']==0, 'lr_abs_mean_mean'], color='blue')
	ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==2, 'lr_abs_mean_mean'],ls='', marker='.', color='green', label='Bass Drum')
	ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==1, 'lr_abs_mean_mean'],ls='', marker='.', color='red', label='Snare Drum / Toms')
	ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==0, 'lr_abs_mean_mean'],ls='', marker='.', color='blue', label='Hi-Hat / Soft Snare Hits')
	ax.legend(loc='lower left', bbox_to_anchor=(.58, 1), fontsize='medium')
	st.subheader('Before/After Audio File Transformation')
	col1, col2 = st.beta_columns(2)
	with col1:
		st.write('')
		st.pyplot(fig1, use_container_width=True)
	with col2:
		st.pyplot(fig5, use_container_width=True)
	st.write('')


def plot_meta_data_graphs(plot_data_dict, expander):
	with expander:
		plot12_data_df = plot_data_dict['plot12']
		plot3_data_df = plot_data_dict['plot3']
		plot4_data_df = plot_data_dict['plot4']
		plot5_data_df = plot_data_dict['plot5']

		# Plot 1: initial pull
		fig1, ax = plt.subplots()
		fig1.suptitle('Original', fontsize=20)
		ax.plot(plot12_data_df['lr'])
		ax.set_xlabel('Hz')
		ax.set_ylabel('Volume')
		# Plot 2: abs value of initial pull
		fig2, ax = plt.subplots()
		fig2.suptitle('Step 1', fontsize=20)
		ax.plot(plot12_data_df['lr_abs'])
		ax.set_xlabel('Hz')
		ax.set_ylabel('Volume')
		# Plot 3: smoothed out (mean of each 100 rows)
		fig3, ax = plt.subplots()
		fig3.suptitle('Step 2', fontsize=20)
		ax.plot(plot3_data_df['lr_abs_mean'])
		ax.set_xlabel('kHz')
		ax.set_ylabel('Volume')
		# Plot 4: mean of each "note" (grouping of numbers with no 0s in between)
		fig4, ax = plt.subplots()
		fig4.suptitle('Step 3', fontsize=20)
		ax.plot(plot4_data_df['lr_abs_mean_mean'])
		ax.set_xlabel('kHz')
		ax.set_ylabel('Volume')
		# Plot 5: plot of each "note" and which volume it got clustered into
		fig5, ax = plt.subplots()
		fig5.suptitle('Step 4', fontsize=20)
		ax.plot(plot5_data_df['lr_abs_mean_mean'],ls='')
		# ax.bar(data_df.loc[data_df['cluster_label']==0].index, data_df.loc[data_df['cluster_label']==0, 'lr_abs_mean_mean'], color='blue')
		ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==2, 'lr_abs_mean_mean'],ls='', marker='.', color='green', label='Bass Drum')
		ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==1, 'lr_abs_mean_mean'],ls='', marker='.', color='red', label='Snare Drum / Toms')
		ax.plot(plot5_data_df.loc[plot5_data_df['cluster_label']==0, 'lr_abs_mean_mean'],ls='', marker='.', color='blue', label='Hi-Hat / Soft Snare Hits')
		ax.legend(loc='lower left', bbox_to_anchor=(.58, 1), fontsize='medium')
		ax.set_xlabel('kHz')
		ax.set_ylabel('Volume')
		st.subheader('Full Audio Clip: Step-by-Step Transformation')
		col1, col2 = st.beta_columns(2)
		with col1:
			st.pyplot(fig1)
			st.pyplot(fig3)
			st.pyplot(fig5)
		with col2:
			st.pyplot(fig2)
			st.pyplot(fig4)
		st.write('')


def plot_single_note_graphs(plot_data_dict, expander):
	with expander:
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

		# Plot 1: initial pul
		fig1, ax = plt.subplots()
		fig1.suptitle('Original', fontsize=20)
		ax.plot(plot12_df.loc[(start-50)*100:(end+50)*100, 'lr'])
		ax.set_xlabel('Hz')
		ax.set_ylabel('Volume')
		# Plot 2: abs value of initial pull
		fig2, ax = plt.subplots()
		fig2.suptitle('Step 1', fontsize=20)
		ax.plot(plot12_df.loc[(start-50)*100:(end+50)*100, 'lr_abs'])
		ax.set_xlabel('Hz')
		ax.set_ylabel('Volume')
		# Plot 3: smoothed out (mean of each 100 rows)
		fig3, ax = plt.subplots()
		fig3.suptitle('Step 2', fontsize=20)
		ax.plot(plot3_df.loc[start-50:end+50, 'lr_abs_mean'])
		ax.set_xlabel('kHz')
		ax.set_ylabel('Volume')
		# Plot 4: mean of each "note" (grouping of numbers with no 0s in between)
		fig4, ax = plt.subplots()
		fig4.suptitle('Step 3', fontsize=20)
		ax.plot(plot4_df.loc[first_note_in_view-1:last_note_in_view+1, 'lr_abs_mean_mean'])
		ax.set_xlabel('kHz')
		ax.set_ylabel('Volume')
		# Plot 5: plot of each "note" and which volume it got clustered into
		fig5, ax = plt.subplots()
		fig5.suptitle('Step 4', fontsize=20)
		plot5_df_low = plot5_df.copy()
		plot5_df_med = plot5_df.copy()
		plot5_df_high = plot5_df.copy()
		plot5_df_low.loc[plot5_df_low['cluster_label']!=0, 'lr_abs_mean_mean'] = -2000
		plot5_df_med.loc[plot5_df_med['cluster_label']!=1, 'lr_abs_mean_mean'] = -2000
		plot5_df_high.loc[plot5_df_high['cluster_label']!=2, 'lr_abs_mean_mean'] = -2000
		ax.plot(plot5_df_high.loc[first_note_in_view-1:last_note_in_view+1, 'lr_abs_mean_mean'],ls='', marker='.', ms=10, color='green', label='Bass Drum')
		ax.plot(plot5_df_med.loc[first_note_in_view-1:last_note_in_view+1, 'lr_abs_mean_mean'],ls='', marker='.', ms=10, color='red', label='Snare Drum / Toms')
		ax.plot(plot5_df_low.loc[first_note_in_view-1:last_note_in_view+1, 'lr_abs_mean_mean'],ls='', marker='.', ms=10, color='blue', label='Hi-Hat / Soft Snare Hits')
		ax.set_ylim([0,highest_vol])
		ax.set_xlabel('kHz')
		ax.set_ylabel('Volume')
		ax.legend(loc='lower left', bbox_to_anchor=(1.03, 1.22), fontsize='medium')
		st.subheader('Zoomed View on First Note(s): Step-by-Step Transformation')
		col1, col2 = st.beta_columns(2)
		with col1:
			st.pyplot(fig1)
			st.pyplot(fig3)
			st.pyplot(fig5)
		with col2:
			st.pyplot(fig2)
			st.pyplot(fig4)
		st.write('')


##################### execution ###########################

title_container = st.beta_container()
title_container.title('Classifying the Genre of a Drum Beat With Machine Learning')
title_container.write('')
title_container.write('')
table_of_contents = """**Table Of Contents**
- Introduction
- Different Genre "Stereotypes"
- View the Results
- Data Transformation
    - Overview
    - Step-By-Step Explanation
    - Graphs
- Comparing Meta Data From Each Genre
    - Graphs
"""
intro_blurb = """Drumming, like all musical instruments, has common "stereotypes" that change for each genre of music.
Using singing as an example, if you were to turn on a country song, you would expect to hear a southern twang in the singer's voice and some long held-out notes. But
if you were to turn on a rap song, you would expect to hear fast and rhythmic singing. These common "stereotypes" exist for drum beats as well. Rock beats are loud with a lot of space between notes,
 while Latin grooves are fast and full of finesse.

 My goal with this personal project was to create a machine learning model that could intake an audio recording from my electronic drumset and classify the genre of the drum beat I played.
 I recorded 120 grooves on my electronic drumset, and did my best to capture the common stereo types of 4 different genres of music:
 Rock, Shuffle, Latin, and Jazz. For each recording, I played 4 measures of a drum beat at varying speeds which resulted in about 9-15 seconds of sound per audio file.
  My electronic drumset records at 44.1 khz which translates into 44.1k rows of data per second of audio (396.9k-661.5k rows per audio file). After transforming each audio file into meta data,
 an XGBoosted classification model was trained to be able to correctly identify the genre of each drum beat at a 90% accuracy rating."""

genre_stereotypes = """The general stereotypes/motifs you can expect in each genre are as follows:
- *Rock:* Big strong hits on the snare and bass drum, with consistent and generally large amounts of space between each note. The hi-hat fills in the space between the drum hits.
- *Shuffle/Funk:* The snare drum contains loud hits with a notable amount of "soft" hits inbetween. More bass drum hits compared to rock. Lots of hi-hat notes that vary from soft to loud.
- *Latin:* The bass drum follows an ever-constant "heart beat" pattern. A "cha-cha" pattern is played around the snare drum and toms. Fast and consistent notes are played on the cymbals. The hi-hat sports a constant "chick" (think boom-chick-boom-chick) pattern.
- *Jazz:* There are little to no bass drum hits. Snare drum hits seem to be played at random and are meant as a form of embellishment, varying from soft to loud. A constant shuffle-like pattern is played on the cymbal with the hi-hat being played on the backbeat
 (where you would normally expect to hear a snare drum in any rock beat)."""

high_level_data_transformation_explanation = """The main goal of transforming the audio file data is to get it in a state to be able to capture the following information:
- 1) Pick out the individual notes that were played during the timeseries soundwave data.
- 2) Calculate the volume of each note and categorize them into 3 buckets: High, Medium, and Low Volume.
- 3) Calculate the number of notes within each Volume Category (High/Medium/Low) and the relationship between these categories.

Being able to extract these main components of meta data from each audio file, gave the model what it needed to be able to detect "stereotypes" within each audio file and ultimately classify the drum beat to its appropriate genre. Examples of these "stereotypes" per genre are listed above.

However, two main factors made transforming this data very tricky.
- First, since the data is in fact audio data, that means each file containts a single table of continuous timeseries data made up of sounds waves. Sound waves are difficult to sift through in data form
 because they oscillate around 0. Initally, I thought I could use a volume value of 0 as the way to know when a note ends, but that was not the case. Because sound waves are constantly oscillating, there are 0s scattered all over the timeseries data; between and *within* the sound of a drum hit.
 - Secondly,  there is *a lot* of data per audio file. On average, each file is about a half a million rows of data. Because of this, I had to get creative in the methods I used to transform the data to avoid hours of processing for each audio file (and to avoid crashing my computer)."""

step_by_step_data_transformation_explanation = """Below is a step-by-step breakdown of the data transformation process. Each step coincides with the graphs shown above.
- 1) The abosulte value was taken to transform all volume data into positive values.
- 2) Each 100 rows of data was grouped up and the average volume was calculated. This was done to get rid of small pockets of 0s that exist within each note as the sound wave oscillated back and forth across 0.
This also had a positive side effect of reducing the total number of rows in each data set from ~500k to ~5k.
- 3) The average volume of each note was then calculated. Because of the calculations done in Step 2, individual notes could now be identified in the timeseries data by using the remaining groups of 0s as the ending point of notes
(ie. volume at 0 == no sound).
- 4) Lastly, each note was classified as either a high-volume note, a medium-volume note, or a low-volume note via KMeans Clustering. By grouping the notes into these volume categories, it allowed for the creation of meta data like
"all of the loud notes are consistently spaced and less frequent (rock)" or "there are a lot of medium and low volume notes (shuffle/funk)". """
with title_container.beta_expander('Introduction', expanded=True):
	st.write(intro_blurb)
with title_container.beta_expander('Genre "Stereotypes"', expanded=True):
	st.write(genre_stereotypes)
	st.write('')
	st.write('Drumset Legend:')
	st.image('https://s3.amazonaws.com/drumeoblog/beat/wp-content/uploads/2019/11/22114343/blog-graphics-labeled-drum-kit.jpg')
title_container.write('Choose a file on the left side of the screen, then listen to the groove and guess the genre! View the results below to see if you and the ML model guessed the same.')


# choosing file to investigate
genres = ['rock', 'shuffle', 'latin', 'jazz']
filename_options = ['{}_{} (used to train/test)'.format(genre, num+1) for genre in genres for num in range(30)]
filename_options = ['drumbeat_{}'.format(num+1) for num in range(8)] + filename_options
st.sidebar.write('**Chose an audio file**')
wav_filename = st.sidebar.selectbox('', filename_options)
if wav_filename == '':
	st.stop()
wav_filename = wav_filename.replace("'", "").replace(' (used to train/test)', '')
wav_filename = wav_filename
# display audio
audio_file = open('groove_samples/' + wav_filename + '.WAV', 'rb')
audio_bytes = audio_file.read()
st.sidebar.audio(audio_bytes, format='audio/ogg')
# table of contents
st.sidebar.write('')
st.sidebar.write(table_of_contents)
# load previously stored results and data
with open('allbeats_data_dict.pkl', 'rb') as handle:
    allbeats_data_dict = pickle.load(handle)
correct_genre = allbeats_data_dict[wav_filename]['correct_genre']
predicted_genre = allbeats_data_dict[wav_filename]['predicted_genre']
single_file_data_df = allbeats_data_dict[wav_filename]['single_file_data_df']
single_file_data_df['genre'] = wav_filename
# running create_meta_data again to get plot_data_dict
_, plot_data_dict, __ = create_meta_data.create_meta_data(wav_filename)

# showing results
with title_container.beta_expander('View the Results!'):
	st.subheader('Audio File: `{}`'.format(wav_filename))
	st.subheader('Correct Genre: `{}`'.format(correct_genre.capitalize()))
	st.subheader('The Model Chose: `{}`'.format(predicted_genre.capitalize()))
title_container.write('')

# data transformation
st.header('Data Transformation')
plot_title_graphs(plot_data_dict)
with st.beta_expander('Data Transformation: Overview'):
	st.write(high_level_data_transformation_explanation)
with st.beta_expander('Data Transformation: Step-By-Step'):
	st.write(step_by_step_data_transformation_explanation)
expander = st.beta_expander('Data Transformation: Graphs')
plot_meta_data_graphs(plot_data_dict, expander)
plot_single_note_graphs(plot_data_dict, expander)
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
bar_mdata['genre'] = bar_mdata['genre'].replace({'rock':'Rock', 'shuffle':'Shuffle', 'latin':'Latin', 'jazz':'Jazz'})
bar_mdata['genre'] = pd.Categorical(bar_mdata['genre'], ['Rock', 'Shuffle', 'Latin', 'Jazz', wav_filename])
bar_mdata = bar_mdata.sort_values(['genre']).reset_index(drop=True)

# creating the graphs
bargraph_values = {
	'Note Count': {'note_count_low':'Low Vol Count', 'note_count_med':'Med Vol Count', 'note_count_high':'High Vol Count'},
	'Note Count Ratios': {'med_low_note_ratio':'Med Vol Count / Low Vol Count', 'high_low_note_ratio':'High Vol Count / Low Vol Count', 'high_med_note_ratio':'High Vol Count / Med Vol Count'},
	'Space Between Notes': {'avg_space_low':'Space Between Low Vol Notes', 'avg_space_med':'Space Between Med Vol Notes', 'avg_space_high':'Space Between High Vol Notes'},
	'Space Between Notes Ratios': {'med_low_space_ratio':'Med Vol Space / Low Vol Space','high_low_space_ratio':'High Vol Space / Low Vol Space', 'high_med_space_ratio':'High Vol Space / Med Vol Space'}
}
x=bar_mdata['genre']
with st.beta_expander('Graphs'):
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
