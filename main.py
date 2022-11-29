# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import csv
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

def process_tweet(input_data):
    for i in range(0, len(input_data), 1):
        input_data[i] = re.sub(r'\&\w*;', '', input_data[i])
        input_data[i] = re.sub('@[^\s]+', '', input_data[i])
        input_data[i] = re.sub(r'\$\w*', '', input_data[i])
        input_data[i] = input_data[i].lower()
        input_data[i] = re.sub(r'https?:\/\/.*\/\w*', '', input_data[i])
        input_data[i] = re.sub(r'#\w*', '', input_data[i])
        input_data[i] = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', input_data[i])
        input_data[i] = re.sub(r'\b\w{1,2}\b', '', input_data[i])
        input_data[i] = re.sub(r'\s\s+', ' ', input_data[i])
        input_data[i] = [char for char in list(input_data[i]) if char not in string.punctuation]
        input_data[i] = ''.join(input_data[i])
        input_data[i] = input_data[i].lstrip(' ')


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

#training_db = training_db[0:10000]

print('moving training into x y')
for each in training_db:    #move TRAINING sentiment score into y and each into x
    x_training.append(each[5])
    y_training.append(int(each[0]))

print('preprocessing x_training')
process_tweet(x_training)

print('preprocessing y_training')
for each in y_training:
    if each == 4:
        each = 1

print('moving test into x y')
for each in test_db:    #move TEST sentiment score into y and each into x
    x_test.append(each[5])
    y_test.append(int(each[0]))

print('preprocessing x_test')
process_tweet(x_test)

print('preprocessing y_test')
for each in y_test:
    if each == 4:
        each = 1    #ignore 2

clf = Pipeline([('hv', HashingVectorizer(alternate_sign=False, stop_words='english')),
                                         ('tfidf', TfidfTransformer()),
                                         ('clf', MultinomialNB())])

clf.fit(x_training, y_training)

incorrect = 0
correct = 0
positive = 0
negative = 0
neutral = 0

t = 0
f = 0
n = 0
for each_x, each_y in zip(x_test, y_test):
    pred = clf.predict([each_x])
    if pred == each_y:
        correct += 1
        if pred == 4:
            positive += 1
        elif pred == 0:
            negative += 1
    else:
        if each_y == 2:
            neutral += 1
        else:
            incorrect += 1

print("Correct = " + str(correct) + "\nIncorrect = " + str(incorrect) + "\nPositive = " + str(positive) +
      "\nNegative = " + str(negative) + "\nNeutral = " + str(neutral))

val = input("Enter your string tweet.  Max 140 characters. \n> ")
while 1:
    if val == '0':
        break
    arr = [val]
    process_tweet(arr)
    score = clf.predict(arr)
    if score == 4:
        print("Your tweet is predicted to be Positive.\n")
    elif score == 0:
        print("Your tweet is predicted to be Negative.\n")
    else:
        print("Your tweet is predicted to be Neutral.\n")

    val = input("Enter your string tweet.  Max 140 characters. \n> ")
print(clf.predict([x_test2]))



print('end')

