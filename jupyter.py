import numpy as np
import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin1')

df.sample(5)

#data cleaning
#EDA
#Text Processing
#Model Building
#Evaluation
#Improvement
#Website
#Deploy

df.info()

#drop last three column
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

# renaming the columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
df.sample(5)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])

df.head()

#missing values
df.isnull().sum()

#check for duplicate values
df.duplicated().sum()

#remove duplicates
df = df.drop_duplicates(keep='first')

df.duplicated().sum()

df['target'].value_counts()

import matplotlib.pyplot as plt

plt.pie(df['target'].value_counts(), labels = ['ham','spam'], autopct="%0.2f")
plt.show()

# Data is imbalanced

import nltk
nltk.data.find('tokenizers/punkt')

df['num_characters'] = df['text'].apply(len)

#num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

#num of sentences
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

df[['num_characters', 'num_words', 'num_sentences']].describe()

#ham

df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()

#spam

df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()

import seaborn as sns

plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'], color='red')

plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color='red')

sns.pairplot(df, hue='target')

import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns
numeric_df = df.select_dtypes(include=[float, int])

# Create the heatmap with correlation matrix
sns.heatmap(numeric_df.corr(), annot=True)

# Show the plot
plt.show()

## Data Preprocessing
# - lower case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - Stemming

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
            
    return y

transform_text('Hi How Are You 234 34$ Vedant?%')
