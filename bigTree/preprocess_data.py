import pandas as pd
import json
from pandas import json_normalize
from datetime import datetime, timedelta
import os
import requests

# date = str(datetime.date.today())

today = datetime.today().strftime("%Y-%m-%d")

yesterday = datetime.today() - timedelta(days=1)
yesDay = yesterday.strftime('%Y%m%d')
yesDate = yesterday.strftime('%Y-%m-%d')

tomday = datetime.today() + timedelta(days=1)
tomday = tomday.strftime('%Y-%m-%d')

# today = "2022-11-01"
# yesDate = "2022-10-31"


curl =  "http://fetnet.infominer.io:8080/api/?start_tm=" + yesDate + "&end_tm=" + tomday

print(curl)

# s=requests.get(curl, proxies={"http": "http://fetfw.fareastone.com.tw:8080"}).content
# s = json.loads(s)

s=requests.get(curl, proxies={"http": "http://fetfw.fareastone.com.tw:8080"}).content
s = json.loads(s)
df = json_normalize(s) #Results contain the required data

df['date'] = pd.to_datetime(df['tm'], unit='s')
df['date'] = df.date.apply(lambda x: x + pd.DateOffset(hours=8) if x  else '')
df_final = df[['id','date', 'url', 'title','content', 'source', 'location', 'type','tags']]

# data = pd.read_csv(io.StringIO(s.decode('utf-8')), sep=",")
save_path = "../data/"
filename = "data_" + today + ".csv"
path = os.path.join(save_path,filename) 
print(path)
df_final.to_csv(path, index = False)



