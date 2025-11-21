# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle


data = pd.read_csv("spam.csv", encoding ='latin-1')


data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4' ], axis = 1, inplace = True)

print(data.head())

#featues and labels
data["label"] = data['class'].map({'ham':0, 'spam':1})

X = data['message']
y = data['label']

#ENcoding
cv = CountVectorizer()
X = cv.fit_transform(X)

pickle.dump(cv, open('CV_object.pkl','wb'))

#train _test split
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.33, random_state= 42)

#Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test,y_test)
print(model.score(X_test,y_test))


#dump pickle file
pickle.dump(model, open('nlp_model.pkl','wb'))




