import nltk
nltk.download('stopwords')
import re # regular expressions
import string # string operations

from nltk.corpus import stopwords # stopwords
from nltk.stem import PorterStemmer # stemming
from nltk.tokenize import TweetTokenizer # tokenization

import numpy as np

''' 
counts how often a word in the corpus was associated with a positive label (1) or negative (0). Results are put into freqs dictionary where key is (word, ys) tuple and value is the frequency.

INPUT
tweets: a list of tweets
labels: an m x 1 array with the sentiment label of each tweet (0 or 1)

OUTPUT
freqs: a dictionary mapping each (word, sentiment) pair to its frequency
'''
def build_freq(tweets, labels):
    # convert np array into a list
    labelslist = np.squeeze(labels).tolist()
    
    # Use zip to map labels to the respective tweet. Y in this case is eiither '1' or '0'
    freqs = {}
    for y, tweet in zip(labelslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            
            freqs[pair] = freqs.get(pair, 0) + 1
                
    return freqs
    

# Cleans the text, tokenizes it, removes stopwords, and converts words to stems.
def process_tweet(tweet):
    strip_tweet = remove_marks(tweet)
    tweet_tokens = tokenize_tweet(strip_tweet)
    cleaned_tweet = remove_stopwords(tweet_tokens)
    
    processed_tweet = execute_stemming(cleaned_tweet)
    
    return processed_tweet
    
    
def execute_stemming(cleaned_tweets):
    stemmer = PorterStemmer()

    tweets_stem = []

    for word in cleaned_tweets:
        stem_word = stemmer.stem(word)
        tweets_stem.append(stem_word)
        
    return tweets_stem

# remove hyperlinks, retweet text, and hashtags.
def remove_marks(tweet):

    # Remove old style text RT
    tmp_tweet = re.sub(r'^RT[\s]+', '', tweet)

    # Remove hyperlink
    tmp_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tmp_tweet)

    # Remove hashtag (#) sign from the beginning of each word
    rm_marks_tweet = re.sub(r'#','',tmp_tweet)
    
    return rm_marks_tweet

def remove_stopwords(tweet_tokens):
    eng_stopwords = stopwords.words("english")

    # Clean tweet
    cleaned_tweets = []

    # for each word in the token list, remove stop words and punctuation
    for word in tweet_tokens:
        if (word not in eng_stopwords and word not in string.punctuation):
            cleaned_tweets.append(word)
    
    return cleaned_tweets
            

def tokenize_tweet(tweet):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    tweet_tokens = tokenizer.tokenize(tweet)
    
    return tweet_tokens

