#!/usr/bin/env python
# coding: utf-8

# ## Step1：取得資料

# In[1]:


# import date
import datetime
import pandas as pd
from pathlib import Path

d1 = datetime.date.today().strftime("%Y-%m-%d")
# d1 = '2022-01-05'


# In[6]:


# df = pd.read_csv('../data/final_' + d1 + '.csv', sep=',')   
# df_raw = df[['發文者','標題','推回文類別','內容','發文時間','觀看次數','推噓評價','發文者分數','URL','crawel_type']]

df = pd.read_csv('../data/final_' + d1 + '.csv', sep=',') 
df_raw = df  
# df_raw = df[['發文者','標題','推回文類別','內容','發文時間','觀看次數','推噓評價','發文者分數','URL','crawel_type']]


# ## Step2: 取得各家情緒分數

# In[3]:


#!pip install transformers #--use-feature=2020-resolver #requests beautifulsoup4 pandas numpy


# In[4]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification,DistilBertTokenizer, DistilBertModel
import torch
import transformers


# In[7]:


pd.set_option('mode.chained_assignment', None)
df_raw.loc[:,'corpus'] = df['title'].fillna(' ') + ": " + df['content'].fillna(' ')


# In[8]:


conditionFilter = 'and ~(corpus.str.contains("廣告"))'
conditionFET = 'corpus.str.contains("遠傳|FET") '
conditionCHT = 'corpus.str.contains("中華|CHT") '
conditionGT = 'corpus.str.contains("亞太|APGT") '
conditionTWN = 'corpus.str.contains("台哥大|台灣大|TWN") '
conditionTWNS = 'corpus.str.contains("台星|台灣之星") '

df_FET = df_raw.query(conditionFET + conditionFilter , engine='python')
df_CHT = df_raw.query(conditionCHT + conditionFilter , engine='python')
df_GT = df_raw.query(conditionGT + conditionFilter , engine='python')
df_TWN = df_raw.query(conditionTWN + conditionFilter , engine='python')
df_TWNS = df_raw.query(conditionTWNS + conditionFilter , engine='python')


# In[11]:


tokenizer = AutoTokenizer.from_pretrained('../module/bert-base-multilingual-uncased-sentiment')
# tokenizer = AutoTokenizer.from_pretrained('adamlin/bert-distil-chinese')

tokenizer.add_tokens(['遠傳','亞太','中華','台哥大','台灣大哥大','台星','台灣之星','5g','4g','5G','4G'], special_tokens=True)
model = AutoModelForSequenceClassification.from_pretrained('../module/bert-base-multilingual-uncased-sentiment')
model.aux_logits = False
model.resize_token_embeddings(len(tokenizer))


# In[12]:


df_raw.head(3)


# In[14]:


def get_score(com_type, corpus, content):
    
    if content.find(com_type):
        corpus = content 
        #若內文有公司名稱,直接以內文分析, 標題不列入情緒分析
        #若內文沒有公司名稱, 則以主題+內文分析.
        #若只考慮內文,會有多筆都沒評分.
    
    tokens = tokenizer.encode(text=com_type, text_pair=corpus, 
                              return_tensors='pt', add_special_tokens = True)
#     print(tokens[:100])
#     print(tokenizer.convert_ids_to_tokens(tokens.squeeze())[:100])
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

com_arr = ['FET','GT','CHT','TWN','TWNS']
com_name_arr = ['遠傳','亞太','中華','台哥大或台灣大哥大','台星或台灣之星']

for i, v in enumerate(com_arr) :
#     print(com_name_arr[i])
#     print(len(locals()['df_'+v].index))
#     print(pd.DataFrame(locals()['df_'+v]['corpus']).head(1))
    df_raw['score'+v] = " "
    if len(locals()['df_'+v].index) :
        df_raw['score'+v] = pd.DataFrame(locals()['df_'+v]).apply(lambda r : get_score('對'+ com_name_arr[i] +'的看法:',
                                                                                       r['corpus'][:500], r['content'][:500]), axis=1)


# In[16]:


# df_final = df_raw[['發文者','標題','推回文類別','內容','發文時間','觀看次數','推噓評價','發文者分數','URL','crawel_type'
                #    ,'scoreFET','scoreTWN','scoreCHT','scoreGT','scoreTWNS']]
df_final = df_raw                


# In[ ]:


# df_final.to_csv('./../crawler_data/clean_data/data_' + d1 + '.csv', index = False)


# In[17]:


df_final.to_csv('../data/final_' + d1 + '.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:




