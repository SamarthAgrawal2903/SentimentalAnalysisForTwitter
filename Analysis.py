import tweepy
import re
import wordcloud
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob as tx
import pandas as pd

''' Twitter API Credentials '''

consumerKey =  ""      #Insert Your Own Consumer Key
consumerKeySecret = ""        #insert Your Own Consumer Secret Key
AccessKey =  ""   #insert Your Own Access Key
AccessKeySecret =  ""  #Insert Your Own Access Secret Key

""" Creating authentication object"""

auth = tweepy.OAuthHandler(consumerKey,consumerKeySecret)

"""Setting up the access token"""

auth.set_access_token(AccessKey,AccessKeySecret)

# creating The API object with auth information

api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

"""Extracting tweets"""
# i = 1
post = api.user_timeline(screen_name = "BillGate", count = 100, lang = "english", tweet_mode = "extended" ,since = "2020-1-1")
# for tweet in post[0:5]:
#     print(str(i) + ")" + tweet.full_text + "\n")
#     i = i + 1

""" Creating a DataFrame """
df = pd.DataFrame([tweet.full_text for tweet in post] , columns=['Tweets'])
df.head()

"""Cleaning the text"""
def cleanText(text):
   text =  re.sub(r"@[A-Za-z0-9]+", ' ', text)
   text = re.sub(r"#", " ", text)
   text = re.sub(r"RT[\s]+", " ", text)
   text = re.sub(r'https:*?\s?', '', text) # removing hyper link
   text = re.sub(r'http:*?\s?', '', text)  # removing hyper link
   return text


df['Tweets'] = df['Tweets'].apply(cleanText)
# print(df)

"""Creating function for subjectivity"""
def getSub(text):
    return tx(text).sentiment.subjectivity

"""Creating function for Polarity"""
def getPolarity(text):
    return tx(text).sentiment.polarity

"""Creating new two Columns """

df["Subjectivity"] = df["Tweets"].apply(getSub)
df["Polarity"] = df["Tweets"].apply(getPolarity)
# print(df)

def getAnalysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"

df["Analysis"] = df["Polarity"].apply(getAnalysis)
print(df)

""" plotting polarity """

plt.figure(figsize=(8,6))
for i in range(0,df.shape[0]):
    plt.scatter(df["Polarity"][1], df["Subjectivity"][i],color = "Blue")
plt.show()