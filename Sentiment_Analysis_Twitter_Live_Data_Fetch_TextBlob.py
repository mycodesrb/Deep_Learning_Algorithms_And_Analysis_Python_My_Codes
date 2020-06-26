#!/usr/bin/env python
# coding: utf-8

# In[1]:


# I will analyse the sentiments for twitter tweets by fetching the tweets from the app 
# I have used an external library to get the weights for the texts
# Import the required packages
import pandas as pd
import numpy as np
import tweepy # for authentication purpose

# Twitter Authentication credentials 
# Confidential, get these credentials by making an account at www.developer.twitter.com

consumerKey = 'confidential'
consumerSecretKey = 'confidential'
accessToken = 'confidential'
accessTokenSecret = 'confidential'

# Authenticate to twitter by creating authenticate object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecretKey, callback=None )

#set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)

# create the API object by passing the authentication info
api = tweepy.API(authenticate, wait_on_rate_limit = True)

# Get 100 public tweets of any twitter user, for Example : Donald Trump
posts = api.user_timeline(screen_name="realDonaldTrump", count=100, lang="en", tweet_mode='extended') 

# Check the last 5 tweets of user's account
print("Last 5 tweets of Arvind Kejriwal :\n")
i=1
for tweet in posts[:5]:
    print(str(i)+": ",tweet.full_text,"\n")
    i+=1

# Make dataframe for tweets
df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
df.head()

df.shape

import re

# Clean the data: remove @txt, #tags, http(s), links and and RT(retweets) : use regular expression
def clean_text(txt):
    text = re.sub(r'@[A-Za-z0-9]+', '', txt) #remove @text
    text = re.sub(r'#', '', text) #remove #tags
    text = re.sub(r'https?:\/\/\S+', '', text) # remove hyperlinks
    text = re.sub(r'RT[\s]+','',text) #\s for one or more white spaces
    
    return text

# Apply on tweets
df['Clean_Tweets'] = df['Tweets'].apply(clean_text)
df

from textblob import TextBlob

# Analyse Subjectivity and polarity:
#Subjectivity: Subjective sentence expresses some personal feelings, views, or beliefs.
# Polarity: Whether the sentence is Positive, Negative or Neutral
def get_Subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

df['Subjectivity'] = df['Clean_Tweets'].apply(get_Subjectivity)
df['Polarity'] = df['Clean_Tweets'].apply(get_polarity)

df.head()

# Try to plot with wordcloud: fun
from wordcloud import WordCloud, STOPWORDS
stop_words = STOPWORDS            

#remove stop words
def rem_stp(txt):
    ls_stop=[]
    for i in txt.split(" "):
        if i not in stop_words:
            ls_stop.append(i)
    return " ".join(ls_stop)

df['Clean_Tweets'] = df['Clean_Tweets'].apply(lambda x: rem_stp(x.lower()))
df.head()

# Try some plotting
import seaborn as sb
import matplotlib.pyplot as plt

sb.distplot(df['Subjectivity'])

sb.distplot(df['Polarity']) #Mostly >0, indicates most positive tweets are between polarity of 0 to 0.1 approx.

text_words=[]
for t in df['Clean_Tweets']:
    text_words += t.split(" ")
    
text_words_str = str(text_words)

#stopword_s = STOPWORDS 
wordCloud = WordCloud(width=500, height=500, background_color='white',
                      random_state=21, min_font_size=10).generate(text_words_str)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordCloud, interpolation="nearest", aspect="auto")
plt.axis('off')
plt.show()

# Analyze Positive, Negative and neutral sentiments
def get_Analysis(p_score):
    if p_score < 0:
        return 'Negative'
    elif p_score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(get_Analysis)    
df.head(10)

sb.countplot(df['Analysis'])

# All positive tweets
df_Positive = df[df['Analysis']=='Positive']
df_Positive.shape

df_Positive.head(10)

# All Negative tweets
df_Negative = df[df['Analysis']=='Negative']
df_Negative.shape

df_Negative.head()

# All Neutral tweets
df_Neutral = df[df['Analysis']=='Neutral']
df_Neutral.shape

df_Neutral.head(10)

# Plotting Analysis
for i in df['Analysis'].unique():
    dfc = df[df['Analysis']==i]
    plt.scatter(dfc['Polarity'], dfc['Subjectivity'], label=i)
plt.title("Polarity vs Subjectivity")
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.legend()
plt.show()

df['Analysis'].unique()

# Percentage of Tweet Category
print("Positive Percentage is : ", round((df_Positive.shape[0]/df.shape[0])*100), "%")
print("Negative Percentage is : ", round((df_Negative.shape[0]/df.shape[0])*100),"%")
print("Neutral Percentage is : ", round((df_Neutral.shape[0]/df.shape[0])*100), "%")

# Here, we can also include other external libraries analyse w.r.t the libraries as they have different logic to derive 
# different sentiments. Analysing all the outputs by the perspective of different libraries, we can consider the best suited
# one for our text data.
