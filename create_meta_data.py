

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

    data_df, genre, plot12_data_df = read_and_format_wav_file(wav_filename)

    data_df = smooth_out_volume(data_df)

    data_df, plot3_data_df, plot4_data_df = calculate_average_volume_per_note(data_df)

    #normalizing the data
    data_df['lr_abs_mean_mean_z'] = (data_df['lr_abs_mean_mean'] - data_df['lr_abs_mean_mean'].mean())/data_df['lr_abs_mean_mean'].std()

    data_df, tot_time, plot5_data_df = cluster_notes_into_volume_levels(data_df)

    #calculating meta data
    note_count_high, note_hz_high, avg_space_high, high_vol_avg = calc_note_stats(data_df, 2, tot_time)
    note_count_med, note_hz_med, avg_space_med, med_vol_avg = calc_note_stats(data_df, 1, tot_time)
    note_count_low, note_hz_low, avg_space_low, low_vol_avg = calc_note_stats(data_df, 0, tot_time)

    high_med_note_ratio = note_count_high/note_count_med
    high_med_space_ratio = avg_space_high/avg_space_med

    high_low_note_ratio = note_count_high/note_count_low
    high_low_space_ratio = avg_space_high/avg_space_low

    med_low_note_ratio = note_count_med/note_count_low
    med_low_space_ratio = avg_space_med/avg_space_low

    med_vol_scale = (med_vol_avg - low_vol_avg) / (high_vol_avg - low_vol_avg)

    plot_data_dict = {'plot12':plot12_data_df, 'plot3':plot3_data_df, 'plot4':plot4_data_df, 'plot5':plot5_data_df}

    #add all meta data for single groove to a row
    row = [note_count_high, note_hz_high, avg_space_high,
                 note_count_med, note_hz_med, avg_space_med,
                 note_count_low, note_hz_low, avg_space_low,
                 high_med_note_ratio, high_med_space_ratio,
                 high_low_note_ratio, high_low_space_ratio,
                 med_low_note_ratio, med_low_space_ratio,
                 med_vol_scale, genre, plot_data_dict, wav_filename]

    return row





def read_and_format_wav_file(wav_filename):
    wav_filename = wav_filename.replace('.wav', '').replace('.WAV', '') + '.WAV'
    genre = re.split('_', wav_filename)[0]

    rate, data = wavfile.read('groove_samples/' + wav_filename)


    # if stereo
    if data.shape[1] == 2:
        data_df = pd.DataFrame(data, columns = ['l', 'r'])
        data_df['lr'] = (data_df['l'] + data_df['r'])/2
    else:
        data_df = pd.DataFrame(data, columns = ['lr'])

    #make all numbers >=0
    data_df['lr_abs'] = abs(data_df['lr'])

    data_df['original_length'] = data_df.shape[0]

    plot12_data_df = data_df.copy()

    return data_df, genre, plot12_data_df




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

    return data_df




def calculate_average_volume_per_note(data_df):
    #average values for each 'note' (groupings of values > 0 separated by values of 0)
    data_df['counter'] = create_counter_col(data_df, 'lr_abs_mean')
    plot3_data_df = data_df.copy()
    data_df['lr_abs_mean_mean'] = data_df.groupby(['counter'])['lr_abs_mean'].transform('mean')
    data_df = data_df[['counter', 'lr_abs_mean_mean']].drop_duplicates().reset_index(drop=True)

    plot4_data_df = data_df.copy()

    return data_df, plot3_data_df, plot4_data_df

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


# grouping up low notes that really should just be one single note
#   However, the were split up due to grouping being 100 which was better for high & med notes
def create_special_low_notes_counter_col(df, col_name):
    counter_col = []
    counter = - 1
    low_note_counter = 0
    inbetween = 0
    for i in range(0, len(df)):
        if (inbetween > 0) and (inbetween <17):
            if (df[col_name][i] == 0):
                counter += 1
                counter_col.append(low_note_counter)
                inbetween += 1
                # st.write(low_note_counter, inbetween, counter)
            elif (df[col_name][i] in [1,2]):
                counter += 1
                counter_col.append(counter)
                low_note_counter = counter + 1
                inbetween = 0
            else:
                counter += 1
                counter_col.append(counter)
                inbetween += 1
        else:
            if (df[col_name][i] == 0):
                counter += 1
                counter_col.append(low_note_counter)
                inbetween += 1
            elif df[col_name][i] in [1,2]:
                counter += 1
                counter_col.append(counter)
                low_note_counter = counter + 1
                inbetween = 0
            else:
                counter += 1
                counter_col.append(counter)
                low_note_counter = counter + 1
                inbetween = 0
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

    # grouping up low notes that really should just be one single note
    #   However, the were split up due to grouping being 100 which was better for high & med notes
    data_df['low_note_counter'] = create_special_low_notes_counter_col(data_df, 'cluster_label')
    grouped_low_note_values = data_df.groupby('low_note_counter')[['lr_abs_mean_mean', 'lr_abs_mean_mean_z']].agg('mean').reset_index(drop=True)
    data_df = data_df.drop_duplicates(['low_note_counter']).reset_index(drop=True)
    data_df[['lr_abs_mean_mean', 'lr_abs_mean_mean_z']] = grouped_low_note_values


    #calculate total groove time (ignoring dead space before and after the groove)
    tot_time = max(all_notes_index) - min(all_notes_index)

    plot5_data_df = data_df.copy()

    return data_df, tot_time, plot5_data_df




def calc_note_stats(df, cluster_label, tot_time):
    note_index = df[df['cluster_label']==cluster_label].index.tolist()
    avg_vol = df.loc[df['cluster_label']==cluster_label, 'lr_abs_mean_mean'].mean()
    spaces = []
    for i in range(1,len(note_index)):
        spaces.append(note_index[i] - note_index[i-1])
    note_count = len(note_index)
    notes_hz = note_count/tot_time
    if note_count > 2:
        avg_space = np.mean(spaces)/np.std(spaces)
    else:
        avg_space = np.nan
    return note_count, notes_hz, avg_space, avg_vol


########## execution #########
