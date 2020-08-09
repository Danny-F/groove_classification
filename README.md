## groove_classification

# Introduction
Being passionate about drumming and data science, I wanted to combine the two into a project. The outcome was a groove_classification model. This model can ingest a drum groove/beat in the form of a .wav file and classify it as either: rock, funk, Latin, or jazz.

95% of the code serves the purpose of manipulating the .wav file and extracting features that the model is able to ingest. Because this was a classification problem, I chose to use the the XGBoost classification model.

The training/test data consists of 120 different drum grooves that are between 10-15 seconds each. I collected the data for this project by playing grooves on my electric drumset, recording 30 grooves for each of the 4 categories (rock, funk, Latin, jazz). My electric drumset records in 44.5 hz which means for each groove I was collected 44.5k rows of data per second of recording.

# Requirements
Python packages:
- pandas
- numpy
- sklearn
- xgboost
- scipy


# Notes
- The create_classifier.py script is used to train and create the model.
- The classify_groove.py script uses the stored model to classify a new drum beat/groove.
