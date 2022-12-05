

import datetime

import numpy as np
import pandas as pd
import os


# In[2]:


base_dir = '../data/'
# base_dir = "/home/jovyan/at102-group4/crawler_data/clean_data"

d1 = datetime.date.today().strftime("%Y-%m-%d")
# d1 = "2022-01-06"
# d1


# In[3]:


d1


# In[4]:


file_name = 'final_'+d1+'.csv'
file_path = os.path.join(base_dir,file_name) 
print(file_path)
df = pd.read_csv(file_path)


# In[5]:


df['title'] = df['title'].astype(str)
df['content'] = df['content'].astype(str)
new_df = pd.DataFrame({'comment_text':df[['title', 'content']].replace('\\n', '', regex=True).agg(''.join, axis=1)})
# new_df = pd.DataFrame({'comment_text':df[['內容']].replace('\\n', '', regex=True).agg(''.join, axis=1)})




import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import BertTokenizer, BertModel, BertConfig


# In[8]:


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# In[9]:


tokenizer = BertTokenizer.from_pretrained('../module/bert-base-chinese')
num_added_toks = tokenizer.add_tokens(['5G','4G','3G','2G','NP','LTE','MB', 'WiFi','2CA','3CA','4CA'])


# In[10]:


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('../module/bert-base-chinese')
        self.l1.resize_token_embeddings(len(tokenizer))
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 5)
    def forward(self, ids, mask, token_type_ids):
#         _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
    
model = BERTClass()
# model.load_state_dict(torch.load('multi_label_1222.pth'))  # https://blog.csdn.net/Avrilzyx/article/details/114586729
model.load_state_dict(torch.load('multi_label_1222_relabel.pth', map_location=torch.device('cpu')))
model.to(device)


# In[53]:


MAX_LEN = 200
VALID_BATCH_SIZE = 4


# In[54]:


tokenizer = BertTokenizer.from_pretrained('../module/bert-base-chinese')


# In[55]:


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
#         self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
#             'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


# In[56]:


testing_set = CustomDataset(new_df, tokenizer, MAX_LEN)


# In[57]:


test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

testing_loader = DataLoader(testing_set, **test_params)


# In[58]:


from tqdm import tqdm


# In[59]:


def validation():
    model.eval()
    fin_outputs=[]
    fin_topk=[]
    with torch.no_grad():
        for _, data in enumerate(tqdm(testing_loader), 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#             targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
#             print(fin_outputs)
#             aaa = torch.sigmoid(outputs).cpu().detach().flatten()
            aaa = torch.sigmoid(outputs).cpu().detach()
#             print(aaa)
#             print(torch.topk(aaa,2))
            fin_topk.extend(torch.topk(aaa,2).indices.numpy().tolist())
#             print(torch.topk(torch.sigmoid(outputs).cpu().detach().flatten(), 2).indices)
    return fin_outputs, fin_topk


# In[60]:


outputs, fin_topk = validation()


# In[61]:


outputs = np.array(outputs)
outputs.shape

fin_topk = np.array(fin_topk)
fin_topk.shape


# In[62]:


outputs2 = np.array(outputs) >= 0.41


# In[63]:


filter_df = pd.DataFrame(outputs2)


# In[64]:


dct = {0: '資費', 1: '網速', 2: '收訊', 3:'5G', 4:'其他'}


# In[65]:


def myfunction(idx, x):
    mylist = []
    if(filter_df.iloc[idx][x[0]]):
        mylist.append(dct[x[0]])
    if(filter_df.iloc[idx][x[1]]):
        mylist.append(dct[x[1]])
    return "/".join(str(x) for x in mylist)


# In[66]:


result = np.array([myfunction(i,v) for i,v in enumerate(fin_topk)])



df['predict_tag'] = pd.DataFrame(result)

final_save_path_file = './../data/'
upload_file_name = 'final_' + d1 + '.csv'
upload_file_path = os.path.join(final_save_path_file,upload_file_name)


upload_file_path


df.to_csv(upload_file_path, index=False)


print("Finish")


import mysql.connector
mydb = mysql.connector.connect(
  host = "10.64.52.133",
  user = "noc",
  password = "noc123",
  database = "FRUIT",
  )
cursor=mydb.cursor()



val = pd.DataFrame(data=None, columns=['brd_id',	'brd_pusher',	'brd_title',	'brd_pushType',	'brd_contents',	'brd_pushTime',	'brd_watchTimes',	'brd_evaluation',	'brd_pusherEvaluation',	'brd_soureUrl',	'brd_sourceWeb',	'brd_articleType',	'brd_fetScore',	'brd_chtScore',	'brd_twnScore',	'brd_gtScore',	'brd_twsScore',	'brd_insertDate',
])

#df = df.astype(object).where(pd.notnull(df), None)

df1 = df.replace({np.nan:None}) #會變成int64,float
df1.info()
df1.scoreFET = df1.scoreFET.astype(str)
df1.scoreCHT = df1.scoreCHT.astype(str)
df1.scoreTWN = df1.scoreTWN.astype(str)
df1.scoreTWNS = df1.scoreTWN.astype(str)
df1.scoreGT = df1.scoreGT.astype(str)


val = []
for a in range(len(df1)):
    
    val.append((df1['id'][a] , '-' , df1['title'][a] , df1['type'][a] , df1['content'][a] , df1['date'][a] , '-' ,'-' ,'-' ,df1['url'][a] ,df1['source'][a] , df1['predict_tag'][a] , df1['scoreFET'][a] ,df1['scoreCHT'][a] , df1['scoreTWN'][a] , df1['scoreGT'][a] , df1['scoreTWNS'][a] ,df1['date'][a].split(" ")[0] ))   # 將d1轉成當天日期:df1['date'][a].strftime('%y-%m-%d')
    
    # val.expand({
    #                 'brd_id': df['id'] ,
    #                 'brd_pusher': '' ,
    #                 'brd_title': df['title'] ,
    #                 'brd_pushType': df['type'] ,
    #                 'brd_contents': df['content'] ,
    #                 'brd_pushTime': df['date'] ,
    #                 'brd_watchTimes': '' ,
    #                 'brd_evaluation': '' ,
    #                 'brd_pusherEvaluation': '' ,
    #                 'brd_soureUrl': df['url'] ,
    #                 'brd_sourceWeb': df['source'] ,
    #                 'brd_articleType': df['predict_tag'] ,
    #                 'brd_fetScore': df['scoreFET'] ,
    #                 'brd_chtScore': df['scoreCHT'] ,
    #                 'brd_twnScore': df['scoreTWN'] ,
    #                 'brd_gtScore': df['scoreGT'] ,
    #                 'brd_twsScore': df['scoreTWNS'] ,
    #                 'brd_insertDate': d1 
    #     } , ignore_index=True)

#https://www.runoob.com/python3/python-mysql-connector.html
sql = "INSERT INTO fruit.fruit_test(brd_id,	brd_pusher,	brd_title,	brd_pushType,	brd_contents,	brd_pushTime,	brd_watchTimes,	brd_evaluation,	brd_pusherEvaluation,	brd_soureUrl,	brd_sourceWeb,	brd_articleType,	brd_fetScore,	brd_chtScore,	brd_twnScore,	brd_gtScore,	brd_twsScore,	brd_insertDate) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
mycursor = mydb.cursor()

for a in range(len(val)):
    try:
        mycursor.execute(sql, val[a])
        mydb.commit() 
    except Exception as e:
        print(e)


# 批量新增
# mycursor = mydb.cursor()
# mycursor.executemany(sql, val)
# mydb.commit()    # 数据表内容有更新，必须使用到该语句
print(mycursor.rowcount, "记录插入成功。")
  
  
