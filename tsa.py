import pandas as pd # Used for working with dataframes
import numpy as np # Used for numerical analysis
import seaborn as sns # used for plotting graphs
import matplotlib.pyplot as plt # used for plotting graphs
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

def message_cleaning(message):
    message_punc_removed = [char for char in message if char not in string.punctuation]
    message_punc_removed_join = ''.join(message_punc_removed)
    message_clean = [word for word in message_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return message_clean

tweets_df = pd.read_csv('data/tweets1.csv')
tweets_df = tweets_df.drop(['id'], axis = 1)
tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)



vectorizer = CountVectorizer(analyzer = message_cleaning)
tweets_countvectorizer = CountVectorizer(analyzer = message_cleaning, dtype = 'uint8').fit_transform(tweets_df['tweet']).toarray()

X = tweets_countvectorizer
Y = tweets_df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)
NB_Classifier = MultinomialNB()
NB_Classifier.fit(X_train, Y_train)

Y_predict_test = NB_Classifier.predict(X_test)

print(classification_report(Y_test, Y_prediction_test))
