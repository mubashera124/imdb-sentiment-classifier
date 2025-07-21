#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import pandas as pd
import re
import string
import swifter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.naive_bayes import MultinomialNB

# Download necessary NLTK resources
# nltk.download('stopwords')
# nltk.download('wordnet')


# In[2]:


#Loading IMDD dataset
df = pd.read_csv('IMDB Dataset.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


# looking at any random review to get insights like: what type of data cleaning are required
df['review'][3]


# ### Text Preprocessing

# In[6]:


# Converting reviews to lowercase 
df['review']=df['review'].str.lower()


# In[7]:


df.head()


# In[8]:


# Creating a text cleaning function

def cleaning_text(text):
    
    
    # removing html tag
    text= re.sub(r'<.*?>','',text)
    
    # removing punctuation
    text=text.translate(str.maketrans('','',string.punctuation))
    
    # converting to lowercase
    text = text.lower()
    
    # removing non-ascii 
    text=re.sub(r'[^\x00-\x7F]','',text)

    # removing numbers
    text=re.sub(r'\d+','',text)
    
    # Remove words with repeated characters like "aaaaahhhhhh"
    text = re.sub(r'\b[a-zA-Z]*([a-z])\1{2,}[a-z]*\b', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # tokenize
    words=text.split()
    
    # removing stopwords
    stop_words=set(stopwords.words('english'))
    words=[word for word in words if word not in stop_words]
               
    # lemmatizing
    lemmatizer=WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


# In[9]:


# Applying cleaning using swifter for performance boost
df['cleaned_review']=df['review'].swifter.apply(cleaning_text)


# In[10]:


# Checking cleaned data
df.head()


# ###  Feature Engineering with TF-IDF

# In[11]:


# Converting cleaned text into TF-IDF features
tfidf=TfidfVectorizer(max_features=1000)


# In[12]:


X = tfidf.fit_transform(df['cleaned_review'])


# In[13]:


# Displaying features name
tfidf.get_feature_names_out()


# In[14]:


X.shape


# ### Label Encoding

# In[15]:


df['sentiment']


# In[16]:


# Map sentiment to binary labels
df['label']=df['sentiment'].map({'positive':1,'negative':0})


# In[17]:


df['label']


# In[18]:


# Checking label distribution
df['label'].value_counts()


# In[19]:


df.head()


# In[20]:


# Split into train and test sets
x_train,x_test,y_train,y_test=train_test_split(X ,df['label'],test_size=0.2,random_state=42)


# In[21]:


x_train.shape


# In[22]:


x_test.shape


# In[23]:


y_train.shape


# In[24]:


y_test.shape


# ### Model 1: Logistic Regression

# In[26]:


## Initialize and train Logistic Regression model
model=LogisticRegression(max_iter=200)


# In[27]:


model.fit(x_train,y_train)


# In[28]:


# Predicting on test set
y_pred=model.predict(x_test)


# Evaluating performance

# In[29]:


accuracy_score(y_test,y_pred)


# In[30]:


classification_report(y_test,y_pred)


# ### Model 2: Multinomial Naive Bayes

# In[31]:


# Initializing and train Logistic Regression model
model= MultinomialNB()


# In[32]:


model.fit(x_train,y_train)


# In[33]:


# Predicting on test set
y_pred=model.predict(x_test)


# In[34]:


#Evaluating performance
accuracy_score(y_test,y_pred)


# In[35]:


classification_report(y_test,y_pred)


# ***"I trained data on both Logistic Regression and Multinomial Naive Bayes to classify IMDB movie reviews into positive or negative sentiments. Logistic Regression performed slightly better, achieving 86% accuracy compared to 83% from Naive Bayes. This suggests that logistic regression is able to better capture the relationship between TF-IDF features and sentiment labels."***
