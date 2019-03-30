import tweepy
import re
from nltk.tokenize import WordPunctTokenizer
from datetime import datetime

key_file = open("twitter_access_keys.txt","r")
lines = key_file.readlines()
key_file.close()

CONS_KEY = lines[0].rstrip()
CONS_SECRET = lines[1].rstrip()
ACC_TOKEN = lines[2].rstrip()
ACC_SECRET =lines[3].rstrip()

def clean_tweets(tweet):
    user_removed = re.sub(r'@[A-Za-z0-9]+','',tweet.decode('utf-8'))
    link_removed = re.sub('https?://[A-Za-z0-9./]+','',user_removed)
    number_removed = re.sub('[^a-zA-Z]', ' ', link_removed)
    lower_case_tweet= number_removed.lower()
    tok = WordPunctTokenizer()
    words = tok.tokenize(lower_case_tweet)
    clean_tweet = (' '.join(words)).strip()
    return clean_tweet
    
def authentication(cons_key, cons_secret, acc_token, acc_secret):
    auth = tweepy.OAuthHandler(cons_key, cons_secret)
    auth.set_access_token(acc_token, acc_secret)
    api = tweepy.API(auth)
    return api
    
def authentication_default():
    return authentication(CONS_KEY, CONS_SECRET, ACC_TOKEN, ACC_SECRET)
    
def get_timestamp():
    today_datetime = datetime.today().now()
    return today_datetime.strftime('%Y%m%d_%H%M%S')
    
def output_tweets(result):
    for tweet in result:
        cleaned_tweet = clean_tweets(tweet['text'].encode('utf-8'))
        print('Tweet: {}'.format(cleaned_tweet))
        print(tweet['created_at'])
        

def write_tweets(result, filename='tweets'):
    file = open('{0}_{1}.txt'.format(filename,get_timestamp()),'w+')    
    for tweet in result:
        cleaned_tweet = clean_tweets(tweet['text'].encode('utf-8'))
        file.write('{0}, {1} \n'.format(tweet['created_at'],cleaned_tweet))
    file.close()
    
