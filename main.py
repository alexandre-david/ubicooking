# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:44:16 2015

@author: Alex
"""

#############
## IMPORTS ##
#############
import pandas as pd
# import numpy as np
#import nltk
import re
from nltk.stem import WordNetLemmatizer
#from sklearn.svm import LinearSVC
#from sklearn.metrics import classification_report
#import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn
import os

#############
## GLOBAL VARIABLES ##
#############
# Paths
path_main = "/Users/Alex/Documents/Challenges/4_whatscooking"
path_script = os.path.join(path_main, "_script")
path_input = os.path.join(path_main, "_input")
path_output = os.path.join(path_main, "_output")

# File names
train_json = "train.json"
test_json = "test.json"



#############
## Load data ##
#############
traindf = pd.read_json(os.path.join(path_input, train_json))
testdf = pd.read_json(os.path.join(path_input, test_json))



#############
## Preprocessing ##
#############
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       


corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = 0.57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
                             
tfidftr=vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
#vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts)

predictors_tr = tfidftr
targets_tr = traindf['cuisine']

predictors_ts = tfidfts





# Logistic regression
parameters = {'C':(1,5,10,20,30,50)}
clf = LogisticRegression(class_weight='auto')
classifier = grid_search.GridSearchCV(clf, parameters)
classifier=classifier.fit(predictors_tr,targets_tr)
classifier.best_score_ # 0.7867
predictions=classifier.predict(predictors_ts)
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

# Naives Bayes Multinomial


# Random Forest
parameters = {'max_depth':(5,10,15,20,30,40,50)}
rf = RandomForestClassifier(n_estimators=500)
classifier = grid_search.GridSearchCV(rf, parameters)
classifier=classifier.fit(predictors_tr,targets_tr)
classifier.best_score_ # 0.727435
predictions=classifier.predict(predictors_ts)
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)


testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("submission.csv")

                
