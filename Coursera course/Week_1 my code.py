# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:40:53 2019

@author: Cagri
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
import re
from ast import literal_eval
import nltk
from nltk.corpus import stopwords
from collections import Counter, OrderedDict


path = r"C:\Users\Cagri\Google Drive\Self Study\Natural Language Processing - Andreas Muller\Coursera course\natural-language-processing-master\data\\"

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data
    
train = read_data(path + "train.tsv")
validation = read_data(path + 'validation.tsv')
test = pd.read_csv(path + 'test.tsv', sep='\t')

X_test = test["title"].values
X_train, y_train = train["title"].values, train["tags"].values
X_val, y_val = validation["title"].values, validation["tags"].values

punctuation_remove = re.compile("[/(){}\[\]\|@,;]")
bad_words = re.compile("[^0-9a-z #+_]")

stop_words = set(stopwords.words('english'))

def text_prepare(text):
    
    text = text.lower()
    text = re.sub(punctuation_remove, " ", text)
    text = re.sub(bad_words, "", text)
    text = text.split()
    text = [words for words in text if not words in stop_words]

    return text

X_test = [text_prepare(sentences) for sentences in X_test]
X_train = [text_prepare(sentences) for sentences in X_train]
X_val = [text_prepare(sentences) for sentences in X_val]


word_count = []
for sentences in X_train:
    for words in sentences:
        word_count.append(words)

words_counts = Counter(word_count)
words_counts = words_counts.most_common()


tag_count = []

for tags in y_train:
    for tag in tags:
        tag_count.append(tag)

tag_counts = Counter(tag_count)
tag_counts = tag_counts.most_common()


listed_dictionary_words = {}
for index, elements in enumerate(words_counts[:5000]):
    listed_dictionary_words[elements[0]] = index

index_to_words = {v: k for k, v in listed_dictionary_words.items()}


def my_bag_of_words(sentences):

    result_vector = np.zeros(len(listed_dictionary_words.keys()))
    for words in sentences:
        if words in listed_dictionary_words.keys():
            result_vector[listed_dictionary_words[words]] =+1
            
    return result_vector











































