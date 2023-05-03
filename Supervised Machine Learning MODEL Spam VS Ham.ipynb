from wordcloud import WordCloud

import numpy as np 
import pandas as pd 
df = pd.read_csv('../input/email-spam-dataset/completeSpamAssassin.csv')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd 
import numpy as np # Numerical Python library for linear algebra and computations
pd.set_option('display.max_columns', None) # code to display all columns
import matplotlib.pyplot as plt
import seaborn as sns 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from collections import Counter
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
import optuna
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/email-spam-dataset/completeSpamAssassin.csv')


#Intial Analysis
df.head()
df.shape
df.info()

#Null Values?
df.isnull().sum()

#droping the singular Null value
df.dropna(inplace=True)

#dropping redundant column
df.drop(['Unnamed: 0'],axis=1, inplace=True)

#Creating a column of characters, words and sentences in each email

df['char'] = df['Body'].apply(len)
df['words'] = df['Body'].apply(lambda x:len(nltk.word_tokenize(x)))
df['sent'] = df['Body'].apply(lambda x:len(nltk.sent_tokenize(x)))

# break down of Spam Vs HAM Emails in the df
plt.figure(figsize=(16,8))
myexplode = [.2, 0]
plt.pie(df['Label'].value_counts(), labels=['HAM', 'SPAM'], explode = myexplode, autopct='%0.2f%%')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df[df['Label'] == 0]['words'])
sns.histplot(df[df['Label'] == 1]['words'], color='red')
plt.show()

sns.pairplot(df, hue='Label')
plt.show()




def text_preprocessing(text):
    # convert  text to lowercase
    text = text.lower()
    
    # creating list of words in email
    text = nltk.word_tokenize(text)
    
    # removing special charecters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    # copying processed text to text and clearing y to store next steps output
    text = y[:]
    y.clear()
    
    # removing stopwords and punctuation marks
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    # stemming 
    stemmer = SnowballStemmer('english')
    for i in text:
        y.append(stemmer.stem(i))
        
    return " ".join(y)




# applying function to text
df['clean_text'] = df['Body'].apply(text_preprocessing)

# creating word cloud of the spam emails

wc = WordCloud(width=1000, height=500, min_font_size=8, background_color='black')

spam_wc = wc.generate(df[df['Label'] == 1]['Body'].str.cat(sep=' '))

plt.figure(figsize=(20,8))
plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['Label'] == 0]['Body'].str.cat(sep=' '))

plt.figure(figsize=(20,8))
plt.imshow(spam_wc)

# creating a dictionary of ml models 
# (models commented out take too long / dont run)
models = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB(),
    # 'SVC': SVC(kernel='sigmoid', gamma=1.0),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=5),
    'LogisticRegression': LogisticRegression(solver='liblinear', penalty='l1'),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=50, random_state=0),
    # 'AdaBoostClassifier': AdaBoostClassifier(n_estimators=50, random_state=0),
    # 'BaggingClassifier': BaggingClassifier(n_estimators=50, random_state=0),
    # 'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=50, random_state=0),
    # 'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=50,random_state=0),
    'SGDClassifier': SGDClassifier(random_state=0)
    }
tfidf = TfidfVectorizer(max_features=17000)
x = tfidf.fit_transform(df['clean_text']).toarray()
y = df['Label'].values
scoring = ['accuracy', 'precision']
results = {}

# getting each models' accuracy and precision scores
for name, model in models.items():
    cv_results = cross_validate(model, x, y, cv=5, scoring=scoring)
    results[name] = {
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'precision_mean': cv_results['test_precision'].mean()
    }

df_results = pd.DataFrame.from_dict(results, orient='index')
print(df_results)
