
import pip
# pip.main(['install','spacy==2.2.0'])
#pip.main(['install','dill'])
#pip.main(['install','seaborn'])

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import metrics
import re
import string
import spacy
import lime
import sklearn.metrics
import pickle
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import dill

#Load testset

reviews_test = pd.read_csv(r'data/MovieDatensatz6_Testset.csv')

#copy dataframe for dataframe visualization
df_show = reviews_test.copy()
df_show = df_show.drop(columns='sentiment')


reviews_test['label'] = 'positive'
df = reviews_test.copy()
for i, row in df.iterrows():
  if row['sentiment']==0:
    reviews_test['label'][i]='negative'
  if row['sentiment']==1:
    reviews_test['label'][i]= 'neutral'


reviews_test['sentiment'] = pd.to_numeric(reviews_test.sentiment, downcast='integer')

#Load clf
f = open(r'data/classifier/dataCleaning.DT', 'rb')
classifier_DT = pickle.load(f)
f.close()

f = open(r'data/classifier/dataCleaning.SVM', 'rb')
classifier_SVM = pickle.load(f)
f.close()

f = open(r'data/classifier/dataCleaning.NB', 'rb')
classifier_NB = pickle.load(f)
f.close()

f = open(r'data/classifier/dataCleaning.LR', 'rb')
classifier_LR = pickle.load(f)
f.close()

f = open(r'data/classifier/dataCleaning.GB', 'rb')
classifier_GB = pickle.load(f)
f.close()

#load important words
df0_NB = pd.read_csv(r'data/ImpWords/df0_NB.csv')
df1_NB = pd.read_csv(r'data/ImpWords/df1_NB.csv')
df2_NB = pd.read_csv(r'data/ImpWords/df2_NB.csv')

df0_SVM = pd.read_csv(r'data/ImpWords/df0_SVM.csv')
df1_SVM = pd.read_csv(r'data/ImpWords/df1_SVM.csv')
df2_SVM = pd.read_csv(r'data/ImpWords/df2_SVM.csv')

df0_LR = pd.read_csv(r'data/ImpWords/df0_LR.csv')
df1_LR = pd.read_csv(r'data/ImpWords/df1_LR.csv')
df2_LR = pd.read_csv(r'data/ImpWords/df2_LR.csv')

df0_DT = pd.read_csv(r'data/ImpWords/df0_DT.csv')
df1_DT = pd.read_csv(r'data/ImpWords/df1_DT.csv')
df2_DT = pd.read_csv(r'data/ImpWords/df2_DT.csv')

df0_GB = pd.read_csv(r'data/ImpWords/df0_GB.csv')
df1_GB = pd.read_csv(r'data/ImpWords/df1_GB.csv')
df2_GB = pd.read_csv(r'data/ImpWords/df2_GB.csv')

#Preprocessing testdata
# test set preprocessing
# transforms into lowercase, if dataCleaning is not used, use this instead
reviews_test['body'] = reviews_test['body'].apply(lambda x: x.lower()) #transform text to lowercase
reviews_test['body'] = reviews_test['body'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))
reviews_test['body'].head()


# used to delete emojis
#punct = string.punctuation
#print(punct)

# Loading Spacy small model as nlp
#nlp = spacy.load("en_core_web_sm")

# Gathering all the stopwords
#from spacy.lang.en.stop_words import STOP_WORDS
#stopwords = list(STOP_WORDS)
#print(len(stopwords))

# dataCleaning method
#def dataCleaning(sentence):
#  doc = nlp(sentence)
#  tokens = []
#  for token in doc:
#    if token.lemma_ != '-PRON-':
#      temp = token.lemma_.lower().strip()
#    else:
#      temp = token.lower_
#    tokens.append(temp)
#  clean_tokens = []
#  for token in tokens:
#    if token not in punct and token not in stopwords:
#      clean_tokens.append(token)
#  return clean_tokens


#Load vectorizer
# load tfidf-vectorizer
# load pickle
#vec = pickle.load(open(r'data/Vectorizer/vector.pickle.SVM", "rb"))
#test_vectors = vec.transform(reviews_test.body)
#print(test_vectors.shape)

#Load test_vectors (instead of vectorizer)
from scipy import sparse
f = open(r'data/Vectorizer/yourmatrix.npz', 'rb')
test_vectors = sparse.load_npz(f)


#Prediction
y_test = reviews_test['sentiment']
# dictionary_reverse = {0:'negative',1:'neutral',2:'positive'}
# y_test_str = []
# for i in y_test:
#     y_test_str.append(dictionary_reverse[i])
# print(y_test_str)

class_names = np.array(['negative', 'neutral', 'positive'])
explainer = LimeTextExplainer(class_names=class_names)

# import Explainer SVM
with open(r'data/explainer/ExplainerSVM', 'rb') as f:
  exp_SVM = dill.load(f)

# import Explainer DT
with open(r'data/explainer/ExplainerDT', 'rb') as f:
  exp_DT = dill.load(f)

# import Explainer NB
with open(r'data/explainer/ExplainerNB', 'rb') as f:
  exp_NB = dill.load(f)

# import Explainer LR
with open(r'data/explainer/ExplainerLR', 'rb') as f:
  exp_LR = dill.load(f)

# import Explainer GB
with open(r'data/explainer/ExplainerGB', 'rb') as f:
  exp_GB = dill.load(f)
