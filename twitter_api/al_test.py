# For time data
from datetime import datetime, timedelta
# For default twitter authentication
import helpers
import tweepy
# For cleaning, tokenizing tweets
import re
from nltk.tokenize import WordPunctTokenizer

today_datetime = datetime.today().now()
yesterday_datetime = today_datetime - timedelta(days=30)
today_date = today_datetime.strftime('%Y-%m-%d')
yesterday_date = yesterday_datetime.strftime('%Y-%m-%d')

api = helpers.authentication_default()

def search_tweets(keyword, total_tweets):
    today_datetime = datetime.today().now() - timedelta(days=5)
    yesterday_datetime = today_datetime - timedelta(days=1)
    today_date = today_datetime.strftime('%Y-%m-%d')
    yesterday_date = yesterday_datetime.strftime('%Y-%m-%d')
    api = helpers.authentication_default()
    search_result = tweepy.Cursor(api.search, 
                                  q=keyword, 
                                  since=yesterday_date, 
                                  until=today_date,
                                  result_type='recent', 
                                  lang='en').items(total_tweets)
    return search_result

 
def clean_tweets(tweet):
    user_removed = re.sub(r'@[A-Za-z0-9]+','',tweet.decode('utf-8'))
    link_removed = re.sub('https?://[A-Za-z0-9./]+','',user_removed)
    number_removed = re.sub('[^a-zA-Z]', ' ', link_removed)
    lower_case_tweet= number_removed.lower()
    tok = WordPunctTokenizer()
    words = tok.tokenize(lower_case_tweet)
    clean_tweet = (' '.join(words)).strip()
    return clean_tweet
    
keyword = 'apple'
total_tweets = 50
tweets = search_tweets(keyword, total_tweets)
for tweet in tweets:
    cleaned_tweet = clean_tweets(tweet.text.encode('utf-8'))
    print('Tweet: {}'.format(cleaned_tweet))
    print(tweet.created_at)