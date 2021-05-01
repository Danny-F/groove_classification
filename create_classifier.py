# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:34:31 2020

@author: danny
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pickle
import create_meta_data
import code



############################## functions ##############################

def create_classifier(meta_data_df):
    #calculating avg classifier accuracy
    best_classifier = ''
    highest_accuracy = 0
    accuracies = []
    for run in range(0,100):
        test_index = np.random.randint(120,size=15)
        #creating training df
        df_train = meta_data_df.copy()
        df_train = meta_data_df.drop(test_index)
        #creating test df
        df_test = meta_data_df.copy()
        df_test = df_test.loc[test_index, :]
        #assigning features
        features = meta_data_df.columns.difference(['genre']).tolist()

        classifier = xgb.XGBClassifier()
        #fitting, predicting, and checking accuracy of classifier
        classifier.fit(df_train[features], df_train['genre'])
        predictions = classifier.predict(df_test[features])
        accuracy = accuracy_score(df_test['genre'], predictions)
        accuracies.append(accuracy)
        if highest_accuracy < accuracy:
            highest_accuracy = accuracy
            best_classifier = classifier
    return best_classifier, np.mean(accuracies)


def nullify_outliers(df, columns):
	column_zscores = []
	for col in columns:
		values_col = df.loc[~df[col].isnull(), col]
		zscores = (values_col - values_col.mean())/values_col.std()
		df.loc[zscores[np.abs(zscores)>=3].index, col] = np.nan
	return df

############################## execution ##############################

genres = [
    'rock',
    'sixeight',
    'latin',
    'jazz'
          ]
num_per_genre = 30

#create meta data df for all files
rows = []
for genre in genres:
    for num in range(1,num_per_genre+1):
        wav_filename = genre + '_' + str(num) + '.wav'
        row = create_meta_data.create_meta_data(wav_filename)
        rows.append(rows)

columns = ['note_count_high', 'note_hz_high', 'avg_space_high',
           'note_count_med', 'note_hz_med', 'avg_space_med',
           'note_count_low', 'note_hz_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio',
           'med_vol_scale', 'plot_data_dict', 'wav_filename'
            'genre']

meta_data_df = pd.DataFrame(rows, columns=columns)
columns.remove('genre')
meta_data_df = nullify_outliers(meta_data_df, columns)

meta_data_df.to_pickle('meta_data_df.pkl')
meta_data_df = pd.read_pickle('meta_data_df.pkl')

code.interact(local=locals())

columns = [
           'note_count_high', 'avg_space_high',
           'note_count_med', 'avg_space_med',
           'note_count_low', 'avg_space_low',
           'high_med_note_ratio', 'high_med_space_ratio',
           'high_low_note_ratio', 'high_low_space_ratio',
           'med_low_note_ratio', 'med_low_space_ratio',
           'med_vol_scale',
            'genre']
meta_data_df = meta_data_df[columns].copy()

#create classifier
best_classifier, average_accuracy = create_classifier(meta_data_df)
print(average_accuracy)

#save best_classifier model
pickle.dump(best_classifier, open('groove_classifier.pkl.dat', 'wb'))


#TODO: calc avg volume for each set of high,med,low vol notes and put med on scale 1-3
