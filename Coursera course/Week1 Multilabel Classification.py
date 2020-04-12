# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:45:46 2019

@author: Cagri
"""

#import sys
#!{sys.executable} -m pip install metrics

#==============================================================================

import sys
sys.path.append("..")
from common.download_utils import download_week1_resources

download_week1_resources()

from grader import Grader
grader = Grader()


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from ast import literal_eval

#================ ast.literal_eval ===========================================
"""
When to use it.

ast.literal_eval(input()) would be useful if you expected a list
 (or something similar) by the user. For example '[1,2]' would be converted to [1,2].

If the user is supposed to provide a number ast.literal_eval(input()) can be
replaced with float(input()), or int(input()) if an integer is expected.
"""
#==============================================================================

import pandas as pd
import numpy as np

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

path = r"C:\Users\Cagri\Google Drive\Self Study\Natural Language Processing - Andreas Muller\Coursera course\natural-language-processing-master\data\\"

train = read_data(path + "train.tsv")
validation = read_data(path + 'validation.tsv')
test = pd.read_csv(path + 'test.tsv', sep='\t')


X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values

X_train.shape[:]
X_train[0]

#==============================================================================

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

#==============================================================================

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE," ",text)
    text = re.sub(BAD_SYMBOLS_RE,"",text)
    text = text.split()
    text = [word for word in text if word not in STOPWORDS]
    text = " ".join(i for i in text)
    return text

#================= Testing the text_prepare function ==========================
    
etc = r"A[ny te\xt CAN b`e this t]o the way"
etc = etc.lower()
etc = re.sub(REPLACE_BY_SPACE_RE," ",etc)
etc = re.sub(BAD_SYMBOLS_RE,"",etc)
etc = etc.split();
etc = [word for word in etc if word not in STOPWORDS]
etc = " ".join(i for i in etc)

#==============================================================================

def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

#==============================================================================

prepared_questions = []
for line in open(path + 'text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)

text_prepare_results = '\n'.join(prepared_questions)

#==============================================================================

X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]

#==============================================================================

# Dictionary of all tags from train corpus with their counts.
tags_counts = {}
# Dictionary of all words from train corpus with their counts.
words_counts = {}

def words_counts_func(text):
    
    j = []
    test_list = []

    for i in text:
        j.append(i.split(" "))

    for k in j:
        for l in k:
            test_list.append(l)
        

    from collections import Counter

    words_counts = Counter(test_list)
    return words_counts

#******************************************************************************

def tag_counts_func(text):
    
    frequency_counter = []

    for i in text:
        for j in i:
            frequency_counter.append(j)

    tags_counts = Counter(frequency_counter)
    return tags_counts



from collections import Counter

tags_counts = tag_counts_func(train["tags"])
words_counts = words_counts_func(X_train)

#==============================================================================

from collections import defaultdict
# Dictionary of all tags from train corpus with their counts.
tags_counts_a =  defaultdict(int)
# Dictionary of all words from train corpus with their counts.
words_counts_a =  defaultdict(int)

for text in X_train:
    for word in text.split():
        words_counts_a[word] += 1


for tags in y_train:
    for tag in tags:
        tags_counts_a[tag] += 1
        
#==============================================================================
        
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:6000]
DICT_SIZE = 5000
WORDS_TO_INDEX = {p[0]:i for i,p in enumerate(most_common_words[:DICT_SIZE])}
INDEX_TO_WORDS = {WORDS_TO_INDEX[k]:k for k in WORDS_TO_INDEX}
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    
    result_vector = np.zeros(dict_size)
    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    
    return result_vector

#******************************************************************************
    
from scipy import sparse as sp_sparse

X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)

#******************************************************************************

def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print(test_my_bag_of_words())

#==============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    
    tfidf_vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.9, ngram_range = (1,2))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.fit_transform(X_test)
    X_val = tfidf_vectorizer.fit_transform(X_val)

    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_

X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

print("\n","c++" in tfidf_reversed_vocab.values(),"\n","c#" in tfidf_reversed_vocab.values())

#******************************************************************************

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')  
    X_train=tfidf_vectorizer.fit_transform(X_train)
    X_val=tfidf_vectorizer.transform(X_val)
    X_test=tfidf_vectorizer.transform(X_test)
                                       
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_

X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

print("\n","c++" in tfidf_reversed_vocab.values(),"\n","c#" in tfidf_reversed_vocab.values())
      
#==============================================================================

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)

#==============================================================================

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    
    OVR = OneVsRestClassifier(LogisticRegression())
    OVR.fit(X_train, y_train)
    return OVR

#******************************************************************************

classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)


y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)


y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))
    
#==============================================================================
    
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def print_evaluation_scores(y_val, predicted):

    accuracy = accuracy_score(y_val, predicted)
    f1 = f1_score(y_val, predicted, average='weighted')
    roc = roc_auc_score(y_val, predicted)
    roc_precision = average_precision_score(y_val, predicted)
    
    return print("\taccuracy: {}".format(round(accuracy,3)),
                 "\n\tF1: {}".format(round(f1,3)),
                 "\n\tAUC: {}".format(round(roc,3)),
                 "\n\tAUC precision: {}".format(round(roc_precision,3)))

print('Bag-of-words\n')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('\nTfidf\n')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

#==============================================================================

def train_classifier(X_train, y_train, penalty):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    
    OVR = OneVsRestClassifier(LogisticRegression(penalty = penalty))
    OVR.fit(X_train, y_train)
    return OVR

classifier_mybag = train_classifier(X_train_mybag, y_train,"l1")
classifier_tfidf = train_classifier(X_train_tfidf, y_train,"l1")

y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

print('Bag-of-words\n')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('\nTfidf\n')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

#==============================================================================

test_predictions = classifier_mybag.predict(X_test_mybag)
test_pred_inversed = mlb.inverse_transform(test_predictions)

test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))



for index in est.coef_.argsort().tolist()[0][:5]:
    print(index)























