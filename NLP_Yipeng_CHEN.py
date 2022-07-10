# -*- coding: utf-8 -*-
"""
Created on 09/July/2022

@author: Yipeng CHEN
"""

#%%
# Reading data

import os
os.chdir("C:/Users/c4780/Desktop/NLP")
os.getcwd()

import pandas as pd

train_df = pd.read_csv('train_set.csv', sep='\t')
print(train_df.head())

# The first column is the category of the news and the second column is the character of the news.

#%%
# Analysis of data


## Analysis of sentence length

train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

# On average, each sentence consists of 907 characters, 
# with the shortest sentence length being 2 and the longest sentence length being 57921.

# Histogram of char count 

import matplotlib.pyplot as plt

_ = plt.hist(train_df['text_len'], bins=200)
plt.xlabel(' Text char count ')
plt.title("Histogram of char count")
plt.savefig('./text_chart_count.png', dpi=750)
plt.show()

# Most of the sentences are less than 2000 in term of length


# News by category

train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
plt.savefig('./category.png', dpi=750)
plt.show()

# The corresponding relationships between categories and numbers in the dataset are as follows: 
# {'Technology': 0, 'Stocks': 1, 'Sports': 2, 'Entertainment ': 3, 
# 'Current Politics': 4, 'Society': 5, 'Education': 6, 'Finance ': 7, 
# 'Furniture': 8, 'Games': 9, 'Realty': 10, 'Fashion ': 11, 'Lottery': 12, 
# 'Horoscope': 13}

# In the training set, technology makes up the largest part of news, 
# followed by stock news, and the smallest part is horoscope news.


## Character distribution statistics
from collections import Counter

all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

print("len(word_count): ", len(word_count)) # 6869
print("word_count[0]: ", word_count[0]) # ('3750', 7482224)
print("word_count[-1]: ", word_count[-1]) # ('3133', 1)

# A total of 6869 words are included in the training set, 
# with the word numbered 3750 appearing most frequently 
# and the word numbered 3133 appearing least frequently

# Based on how the characters appear in each sentence,
# We can also guess which character is likely to be a punctuation mark 

train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d: int(d[1]), reverse = True)

print("word_count[0]: ", word_count[0]) 
print("word_count[1]: ", word_count[1])
print("word_count[1]: ", word_count[2])

# The code above counts the number of times different characters appearing in a sentence,
# where character 3750, character 900 and character 648 have close to 99% coverage in 200,000 news items.
# These characters are likely to be punctuation marks.



### Conclusion : 
# The imbalance of categories may seriously affect the accuracy of the model

#%%

# Ridge regression is a method of estimating the coefficients of multiple-regression models 
# in scenarios where linearly independent variables are highly correlated.

# Cross-validation is a robust measure to prevent overfitting

# Count Vectorizer is a way to convert a given set of strings into a frequency representation

# TF-IDF means Term Frequency - Inverse Document Frequency. 
# This is a statistic that is based on the frequency of a word in the corpus 
# but it also provides a numerical representation of how important a word is for statistical analysis.

# TF-IDF is better than Count Vectorizers because it not only focuses on the frequency of words present in the corpus, 
# but also provides the importance of the words. We can then remove the words that are less important for analysis, 
# hence making the model building less complex by reducing the input dimensions.

# Let's see if TF-IDF is really better than Count Vectors (Bag of Words) in our case !

# Count Vectors + RidgeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score

train_df = pd.read_csv('train_set.csv', sep='\t', nrows=15000)

vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

clf = RidgeClassifierCV()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.7495

# TF-IDF + RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score

train_df = pd.read_csv('train_set.csv', sep='\t', nrows=15000)

tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = RidgeClassifierCV()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.8719

# 0.8719 > 0.7495 
# => In the condition of using exactly the same classifier, 
# TF-IDF is really better than Count Vectors !

# Therefore, we will focus on the TF-IDF 
# and improve our Ridge regression model for the entire training set !

#%% 
# Data preprocessing

import time 

since = time.time()

import pandas as pd

import os
os.chdir("C:/Users/c4780/Desktop/NLP")
os.getcwd()

train_df = pd.read_csv('train_set.csv', sep='\t') 
test_df = pd.read_csv('test_a.csv')

# concat:
df = pd.concat([train_df,test_df])

print(type(df))

df[199999:]

train_df.head()

test_df.head()

df_text = pd.DataFrame((pd.concat([train_df['text'],test_df['text']])).reset_index(drop=True))

df_text[199999:]

df_text.head() 

# TF-IDF :
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=18000)
tfidf_vector = tfidf.fit_transform(df_text['text']) 

tfidf_vector

tfidf_train = tfidf_vector[0:200000]
tfidf_test = tfidf_vector[200000:]

time_elapsed = time.time() - since
print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


#%%
# Now, we want to build a Machine Learning model that can understand the pattern based on the words present in the strings.

## Text classification based on machine learning models

# RidgeClassifierCV
 
import time 

since = time.time()

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(tfidf_train, train_df['label'], test_size = 0.2, random_state= 18000)

clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1])

clf.fit(X_train, y_train.values) 

val_pred = clf.predict(X_test)

print(f1_score(y_test.values, val_pred, average='macro'))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

pred_clf = clf.predict(tfidf_test)
print(pred_clf)
pred_clf1 = pd.DataFrame({'label':pred_clf}) 

pred_clf1.to_csv('pred_clf.csv',index = False)

xx = pd.read_csv('pred_clf.csv')
xx.info

