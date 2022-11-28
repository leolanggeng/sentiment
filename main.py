# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import csv
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

print('hello world')

training_db = []
test_db = []
x_training = []
y_training = []
x_test = []
y_test = []

print('opening training file')
with open('training.1600000.processed.noemoticon.csv', 'r') as csvfile: #read training file
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        training_db.append(row)

print('opening test file')
with open('testdata.manual.2009.06.14.csv', 'r') as csvfile:    #read test file
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        test_db.append(row)

training_db = training_db[0:10000]

print('moving training into x y')
for each in training_db:    #move TRAINING sentiment score into y and each into x
    x_training.append(each[5])
    y_training.append(int(each[0]))

print('preprocessing x_training')
for i in range(0, len(training_db), 1):
    x_training[i] = re.sub(r'\&\w*;', '', x_training[i])
    x_training[i] = re.sub('@[^\s]+', '', x_training[i])
    x_training[i] = re.sub(r'\$\w*', '', x_training[i])
    x_training[i] = x_training[i].lower()
    x_training[i] = re.sub(r'https?:\/\/.*\/\w*', '', x_training[i])
    x_training[i] = re.sub(r'#\w*', '', x_training[i])
    x_training[i] = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', x_training[i])
    x_training[i] = re.sub(r'\b\w{1,2}\b', '', x_training[i])
    x_training[i] = re.sub(r'\s\s+', ' ', x_training[i])
    x_training[i] = [char for char in list(x_training[i]) if char not in string.punctuation]
    x_training[i] = ''.join(x_training[i])
    x_training[i] = x_training[i].lstrip(' ')

print('preprocessing y_training')
for each in y_training:
    if each == 4:
        each = 1

#vectorizing training
print('vectorizing training')
vectorizer = CountVectorizer()

#encoding all words in the x_training
vectorizer.fit(x_training)
#printing vocab
print("Vocabulary: ", vectorizer.vocabulary_)

print('encode x_training')
vectorized_x = vectorizer.transform(x_training)
# Summarizing the Encoded Texts

print('toarray')
#x_training = vectorized_x.toarray()
x_training = []
for each in vectorized_x:
    x_training.append(each.toarray())


print("Encoded Document is:")
print(x_training[0:3])

print('training model')
clf = ComplementNB()
clf.fit([x_training], y_training)

print('moving test into x y')
for each in test_db:    #move TEST sentiment score into y and each into x
    x_test.append(each[5])
    y_test.append(int(each[0]))

print('preprocessing x_test')
for i in range(0, len(test_db), 1):
    x_test[i] = re.sub(r'\&\w*;', '', x_test[i])
    x_test[i] = re.sub('@[^\s]+', '', x_test[i])
    x_test[i] = re.sub(r'\$\w*', '', x_test[i])
    x_test[i] = x_test[i].lower()
    x_test[i] = re.sub(r'https?:\/\/.*\/\w*', '', x_test[i])
    x_test[i] = re.sub(r'#\w*', '', x_test[i])
    x_test[i] = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', x_test[i])
    x_test[i] = re.sub(r'\b\w{1,2}\b', '', x_test[i])
    x_test[i] = re.sub(r'\s\s+', ' ', x_test[i])
    x_test[i] = [char for char in list(x_test[i]) if char not in string.punctuation]
    x_test[i] = ''.join(x_test[i])
    x_test[i] = x_test[i].lstrip(' ')


t = 0
f = 0
for each_x, each_y in zip(x_test, y_test):
    pred = clf.predict([each_x])
    if (pred == each_y):
        t += 1
    else:
        f += 1

print(t, f)

print('end')