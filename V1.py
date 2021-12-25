# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 19:36:40 2021

@author: moham
"""
#%%
import pandas as pd

df=pd.read_csv('ar_reviews_100k.tsv',delimiter='\t')

df=df[df['label']!='Mixed']
#%%

import numpy as np 
import pandas as pd 
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pylab as plt


from sklearn import preprocessing
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import Counter 
import re
import string
import matplotlib.cm as cm
from matplotlib import rcParams
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
stopwords_list = stopwords.words('arabic')
def train_model(model, data, targets,n=1):
    text_clf = Pipeline([
    ('vect', TfidfVectorizer(min_df=0.0001, max_df=0.95,
                                 
                                 ngram_range=(1, n),stop_words=stopwords_list)),
    ('tfidf', TfidfTransformer()),
    ('clf', model),
    ])
    text_clf.fit(data, targets)
    return text_clf
def get_accuracy(trained_model,X, y):
    predicted = trained_model.predict(X)
    accuracy = np.mean(predicted == y)
    return accuracy

#%%
cats=df['label'].value_counts()

pro= preprocessing.LabelEncoder()
df['label']= pro.fit_transform(df['label'])


X,y=df['text'],df['label']
#%%

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return str(text).translate(translator)
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", str(text))
    text = re.sub("ى", "ي", str(text))
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

#Remove Rpeat Char
def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)
def processRequest(requestStr): 

    
    
    
    # normalize the tweet
    requestStr= normalize_arabic(requestStr)
    
    # remove repeated letters
    requestStr=remove_repeating_char(requestStr)
    
    return requestStr
#%%
X.apply(lambda x: processRequest(x))
#%%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#%%
models = []
models.append(('DecisionTree',DecisionTreeClassifier()))
models.append(('MultinomialNB', MultinomialNB()))
models.append(('LinearSVC', LinearSVC()))
models.append(("xgboost",XGBClassifier()))
models.append(('RandomForest', RandomForestClassifier()))

models.append(('RidgeClassifier',RidgeClassifier()))
models.append(('PassiveAggressive',PassiveAggressiveClassifier()))
def tablengram(n,models,X_train, X_test, y_train, y_test):
    tableResult=PrettyTable()
    print(str(n).center(30,'-'))
    tableResult.field_names=['Model Name','Accurance','Precision','Recall']
    for name, model in models:
        trainModel = train_model(model, X_train, y_train,n)
        accuracy = get_accuracy(trainModel,X_test, y_test)
        print(f"Test dataset accuracy with {name}: {accuracy:.4f}")
        y_pred = trainModel.predict(X_test)
        
        Precision=round(f1_score(y_test
                                         ,y_pred
                                         ,pos_label='positive'
                                         ,average='micro'),2)
        Recall=round(recall_score(y_test, y_pred,pos_label='positive'    ,average='micro'),2)
        tableResult.add_row([name, accuracy, str(Precision)+'%', str(Recall)+'%'])
    print(tableResult)
#%%
for i in np.arange(1,4):
    tablengram(i,models,X_train,X_test,y_train,y_test)
#leanercsv n=2 next step get best prams 
#%%
model=train_model(LinearSVC(), X, y,2)
#%%
    
predict_test=model.predict([	'ممتاز نوعا ما . النظافة والموقع والتجهيز والشاطيء. المطعم', 'أحد أسباب نجاح الإمارات أن كل شخص في هذه الدولة يعشق ترابها. نحن نحب الإمارات. ومضات من فكر. نصائح لدولة تطمح بالصفوف الأولى و قائد لا يقبل إلا براحة شعبه وتوفر كل سب العيش الكريم. حكم و مواقف ونصائح لكل فرد فينا ليس بمجرد كتاب سياسي كما كنت اعتقد. يستحق القراءة مرات كثيرة'])
print(pro.inverse_transform(predict_test))
#%%
import pickle
pickle.dump(model, open('model.sav', 'wb'))
pickle.dump(pro, open('label.sav', 'wb'))
