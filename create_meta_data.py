

import numpy as np
from scipy.io import wavfile
from sklearn.cluster import KMeans
import seaborn as sns
import math
import re
import pandas as pd
from matplotlib import pyplot as plt
import streamlit as st
############################## functions ##############################


def create_meta_data(wav_filename):

    data_df, genre = read_and_format_wav_file(wav_filename)

    data_df = smooth_out_volume(data_df)

    data_df = calculate_average_volume_per_note(data_df)

    #normalizing the data
    data_df['lr_abs_mean_mean_z'] = (data_df['lr_abs_mean_mean'] - data_df['lr_abs_mean_mean'].mean())/data_df['lr_abs_mean_mean'].std()

    data_df, tot_time = cluster_notes_into_volume_levels(data_df)

    #calculating meta data
    note_count_high, note_hz_high, avg_space_high = calc_note_stats(data_df, 2, tot_time)
    note_count_med, note_hz_med, avg_space_med = calc_note_stats(data_df, 1, tot_time)
    note_count_low, note_hz_low, avg_space_low = calc_note_stats(data_df, 0, tot_time)

    high_med_note_ratio = note_count_high/note_count_med
    high_med_space_ratio = avg_space_high/avg_space_med

    high_low_note_ratio = note_count_high/note_count_low
    high_low_space_ratio = avg_space_high/avg_space_low

    med_low_note_ratio = note_count_med/note_count_low
    med_low_space_ratio = avg_space_med/avg_space_low

    #add all meta data for single groove to a row
    row = [note_count_high, note_hz_high, avg_space_high,
                 note_count_med, note_hz_med, avg_space_med,
                 note_count_low, note_hz_low, avg_space_low,
                 high_med_note_ratio, high_med_space_ratio,
                 high_low_note_ratio, high_low_space_ratio,
                 med_low_note_ratio, med_low_space_ratio,
                 genre]
    return row





def read_and_format_wav_file(wav_filename):
    wav_filename = wav_filename.replace('.wav', '') + '.wav'
    genre = re.split('_', wav_filename)[0]

    rate, data = wavfile.read('groove_samples/' + wav_filename)


    # if stereo
    if data.shape[1] == 2:
        data_df = pd.DataFrame(data, columns = ['l', 'r'])
        data_df['lr'] = (data_df['l'] + data_df['r'])/2
    else:
        data_df = pd.DataFrame(data, columns = ['lr'])

    # fig, ax = plt.subplots()
    # ax.plot(data_df['lr'])
    # st.pyplot(fig)


    #make all numbers >=0
    data_df['lr_abs'] = abs(data_df['lr'])

    # fig, ax = plt.subplots()
    # ax.plot(data_df['lr_abs'])
    # st.pyplot(fig)

    data_df['original_length'] = data_df.shape[0]

    return data_df, genre




def smooth_out_volume(data_df):
    #smooth out data (getting rid of random small pockets of 0's)
    group_size = 100
    group_num = [[x]*group_size for x in range(math.ceil(len(data_df)/group_size))]
    group_num = [x for b in group_num for x in b]
    group_num = group_num[0:len(data_df)]
    data_df['group_num'] = group_num
    data_df['lr_abs_mean'] = data_df.groupby(['group_num'])['lr_abs'].transform('mean')
    data_df.loc[data_df['lr_abs_mean']<100, 'lr_abs_mean'] = 0
    data_df = data_df.drop_duplicates(['group_num']).reset_index(drop=True)

    # fig, ax = plt.subplots()
    # ax.plot(data_df['lr_abs_mean'])
    # st.pyplot(fig)

    return data_df




def calculate_average_volume_per_note(data_df):
    #average values for each 'note' (groupings of values > 0 separated by values of 0)
    data_df['counter'] = create_counter_col(data_df, 'lr_abs_mean')
    data_df['lr_abs_mean_mean'] = data_df.groupby(['counter'])['lr_abs_mean'].transform('mean')
    data_df = data_df[['counter', 'lr_abs_mean_mean']].drop_duplicates().reset_index(drop=True)

    # fig, ax = plt.subplots()
    # ax.plot(data_df['lr_abs_mean_mean'])
    # st.pyplot(fig)

    return data_df

def create_counter_col(df, col_name):
    counter_col = []
    counter = 0
    for i in range(len(df)):
        if df[col_name][i] == 0:
            counter_col.append(counter)
            counter += 1
        else:
            counter_col.append(counter)
    return counter_col




def cluster_notes_into_volume_levels(data_df):
    #identifying notes and clustering them into 3 different 'volume' levels
    all_notes_index = data_df[data_df['lr_abs_mean_mean']>0].index.tolist()
    df_notes = data_df.loc[all_notes_index,:].copy()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(np.array(df_notes['lr_abs_mean_mean_z']).reshape(-1,1))
    df_notes['cluster_label'] = kmeans.labels_
    data_df['cluster_label'] = df_notes['cluster_label']
    df_notes['cluster_mean'] = df_notes.groupby(['cluster_label'])['lr_abs_mean_mean_z'].transform('mean')
    #ordering cluster labels to make sure '0' is the lowest volume cluster and '1' the highest
    cluster_rank_list = df_notes['cluster_mean'].unique().tolist()
    cluster_rank_list.sort()
    df_notes.loc[df_notes['cluster_mean']==cluster_rank_list[0], 'cluster_label'] = 0
    df_notes.loc[df_notes['cluster_mean']==cluster_rank_list[1], 'cluster_label'] = 1
    df_notes.loc[df_notes['cluster_mean']==cluster_rank_list[2], 'cluster_label'] = 2
    data_df['cluster_label'] = df_notes['cluster_label']
    #calculate total groove time (ignoring dead space before and after the groove)
    tot_time = max(all_notes_index) - min(all_notes_index)

    # fig, ax = plt.subplots()
    # ax.plot(data_df['lr_abs_mean_mean'],ls='')
    # # ax.bar(data_df.loc[data_df['cluster_label']==0].index, data_df.loc[data_df['cluster_label']==0, 'lr_abs_mean_mean'], color='blue')
    # ax.plot(data_df.loc[data_df['cluster_label']==0, 'lr_abs_mean_mean'],ls='', marker='.', color='blue')
    # ax.plot(data_df.loc[data_df['cluster_label']==1, 'lr_abs_mean_mean'],ls='', marker='.', color='green')
    # ax.plot(data_df.loc[data_df['cluster_label']==2, 'lr_abs_mean_mean'],ls='', marker='.', color='red')
    # st.pyplot(fig)

    return data_df, tot_time




def calc_note_stats(df, cluster_label, tot_time):
    note_index = df[df['cluster_label']==cluster_label].index.tolist()
    spaces = []
    for i in range(1,len(note_index)):
        spaces.append(note_index[i] - note_index[i-1])
    note_count = len(note_index)
    notes_hz = note_count/tot_time
    if note_count > 2:
        avg_space = np.mean(spaces)/np.std(spaces)
    else:
        avg_space = np.nan
    return note_count, notes_hz, avg_space


########## execution #########
