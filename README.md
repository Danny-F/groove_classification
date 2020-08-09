## groove_classification

# Introduction
Being passionate about drumming and data science, I wanted to combine the two into a project. The outcome was a groove_classification model. This model can ingest a drum groove/beat in the form of a .wav file and classify it as either: rock, funk, latin, or jazz.

95% of the code is around manipulating the .wav file and extracting features that the model is able to ingest. Because this was a classification problem, I chose to use the the XGBoost classification model.

The training/test data consists of 120 different drum grooves that are between 10-15 seconds each. The data was collected by me playing grooves on my electric drumset, and I recorded 30 grooves for each of the 4 categories (rock, funk, latin, jazz). My electric drumset records in 44.5 hz which essentially means for each groove I was collecting 44.5k rows of data per second.

# Requirements
Python packages:
- pandas
- numpy
- pickle
- sklearn
- xgboost
- scipy
- math
- re

# Notes
- The create_classifier.py script is used to train and create the model.
- The classify_groove.py script uses the stored model to classify a new drum beat/groove.
