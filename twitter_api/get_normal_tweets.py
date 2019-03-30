from TwitterAPI import TwitterAPI
import helpers

from datetime import datetime, timedelta

#formatting stuff for what time we want tweets
until_datetime = datetime.today().now() - timedelta(days=0)
since_datetime = until_datetime - timedelta(days=1) 
until_date = until_datetime.strftime('%Y-%m-%d')
since_date = since_datetime.strftime('%Y-%m-%d')
    
#get the api    
api = TwitterAPI(helpers.CONS_KEY, helpers.CONS_SECRET, helpers.ACC_TOKEN, helpers.ACC_SECRET, auth_type = 'oAuth1')    

#simple search, don't need fancy stuff for this
endpoint = "search/tweets"
data = {'q': 'apple finance', 'since':since_date,'until':until_date, 'count': '25'}

#get the data
r = api.request(endpoint, data)

#write to a file (second optional argument is filename)
helpers.write_tweets(r, 'apple_finance')
#print that shit out
helpers.output_tweets(r)