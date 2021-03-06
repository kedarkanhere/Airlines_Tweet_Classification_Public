# -*- coding: utf-8 -*-
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

data_file = '../input/Usecase3_Dataset.csv'

data = pd.read_csv(data_file)

###EDA starts here###
data.head()

data.shape

data.info

data.airline.value_counts().plot(kind='bar')

data.airline_sentiment.value_counts().plot(kind='pie',autopct='%1.0f')

##So our dataset is imbalanced. For classification going ahead we can use stratifed cross validation or imblearn to balance data
plt.figure(figsize=(12,7))
sns.countplot(x='airline',hue='airline_sentiment',data=data,palette='rainbow')

airline_tweets=data[data ['airline_sentiment']=='positive']
words = ' '.join(airline_tweets ['text'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])
# Calculate highest frequency words in positive tweets
def freq(str): 
  
    # break the string into list of words  
    str = str.split()          
    str2 = [] 
  
    # loop till string values present in list str 
    for i in str:              
  
        # checking for the duplicacy 
        if i not in str2: 
  
            # insert value in str2 
            str2.append(i)  
              
    for i in range(0, len(str2)): 
        if(str.count(str2[i])>50): 
            print('Frequency of', str2[i], 'is :', str.count(str2[i]))
        
print(freq(cleaned_word))

air_senti=pd.crosstab(data.airline, data.airline_sentiment)
air_senti

percent=air_senti.apply(lambda a: a / a.sum() * 100, axis=1)
percent

#So overall we can see that Virgin america is having the most positive response and US airways is having the worst
#Though we can argue that VA data points are less as compared to Delta


#We can also use N gram finding techniques to see which N grams are present in the text
