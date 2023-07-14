import pandas as pd
import snscrape.modules.twitter as sn
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
from typing import OrderedDict

# path for the original dataset
path = ""

dt = pd.read_csv(path, sep='\t', header=None, names=["date", "user", "text", "label"], encoding='cp1252')
print(dt)

rows, columns = dt.shape

tweets_list = []

for k in tqdm(range(rows)):
    
    username = dt.at[k, "user"]

    text = dt.at[k, "text"]
    index = text.find("#")
    if (index != -1 and index != 0):
        text = text[0:index]

    date = dt.at[k, "date"]
    date = date.split(" ")
    date = date[0]

    date_time_obj = datetime.strptime(date, '%Y-%m-%d')
    date_time_obj += timedelta(days=1)

    date_end = date_time_obj.strftime('%Y-%m-%d')
  
    search = text + " from:" + username + " since:" + date + " until:" + date_end

    try:
        items = sn.TwitterSearchScraper(search).get_items()
        for j,tweet in enumerate(items):
            tweets_list.append([tweet.id, tweet.user.username, tweet.user.id, tweet.user.followersCount, tweet.user.friendsCount, tweet.date, tweet.content])
            print([tweet.id, tweet.user.username, tweet.user.id, tweet.user.followersCount, tweet.user.friendsCount, tweet.date, tweet.content])    
    except:
        print("error")

# path for the new dataset
path_save = ""
path_save = path_save + "tweets_palin.csv"

tweets_df = pd.DataFrame(tweets_list, columns=['Tweet_id', 'Username', 'User_id', 'User_followers', 'User_friends', 'Datetime', 'Text'])
tweets_df.to_csv(path_save, index=False)    

# path of the new tweets dataset
path = ""

dt = pd.read_csv(path)

rows, columns = dt.shape

users_list = []

for i in range(rows):
    id = dt.at[i, "User_id"]
    username = dt.at[i, "Username"]
    followers_count = dt.at[i, "User_followers"]
    friends_count = dt.at[i, "User_friends"]
    users_list.append([id, username, followers_count, friends_count])

# path for the users dataset
path = ""

users_df = pd.DataFrame(users_list, columns=['User_id', 'Username', 'User_followers', 'User_friends'])
users_df = users_df.drop_duplicates()
users_df.to_csv(path, index=False)   

