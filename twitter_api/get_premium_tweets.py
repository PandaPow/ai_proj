from TwitterAPI import TwitterAPI
import helpers

#Using the keys from the helper file
api = TwitterAPI(helpers.CONS_KEY, helpers.CONS_SECRET, helpers.ACC_TOKEN, helpers.ACC_SECRET, auth_type = 'oAuth2')    

#Use dev environment
endpoint = "tweets/search/fullarchive/:dev"

#fromDate/toDate format is yyyymmddhhmm
data = {'query': 'apple finance', 'fromDate':'201802020000','toDate':'201802240000', 'maxResults': '10'}

#get data
r = api.request(endpoint, data)

#write it to a file (second optional argument is filename)
helpers.write_tweets(r,'apple_finance')                
#print that shit to a screen
helpers.output_tweets(r)
