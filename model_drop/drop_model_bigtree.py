from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
# from transformers import DataCollatorWithPadding

from datasets import load_dataset
import pandas as pd
# import numpy as np
import os
import torch
from tqdm.notebook import tqdm

import datetime


tokenizer = AutoTokenizer.from_pretrained('../module/roberta-base-finetuned-chinanews-chinese/')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'



# 載入test資料
import datetime


date = str(datetime.date.today())
date_today = 'data_'+ date + '.csv'

datapath_test_file = "./../data/"
datapath_test = os.path.join(datapath_test_file,date_today) 

df_test_all = pd.read_csv(datapath_test)
date_today


# 製造一份相同格式的測試資料，僅留標題+內文
 
data = df_test_all['title']+'[sep]'+ df_test_all['content']
df_test = pd.DataFrame(data=data[:20],columns=['content'])
df_test['label'] = None
df_test = df_test[['label','content']]
df_test.dropna(how='all', inplace=True)

# 存檔處理後的test
save_path_file = './../data/test/'

save_path = os.path.join(save_path_file,'test_'+date+'.csv')
df_test.to_csv(save_path,index=False)


data=load_dataset('csv', data_files={
                                     'test':[save_path]}) 


def tokenize_function(data:dict):
    return tokenizer(data['content'],padding=True,truncation=True, max_length=512)

tokenized_data=data.map(tokenize_function, batched=True)

tokenized_data = tokenized_data.remove_columns(['content'])
tokenized_data = tokenized_data.rename_column('label','labels')
tokenized_data.set_format('torch',device='cpu')
tokenized_data['test'].column_names
tokenized_data.set_format('torch',device='cpu')


from torch.utils.data import DataLoader

BATCHSIZE = 8

# train_dataloader = DataLoader(
#     tokenized_data["train"], shuffle=True, batch_size=BATCHSIZE, collate_fn=data_collator
# )
# valid_dataloader = DataLoader(
#     tokenized_data["valid"], shuffle=False, batch_size=BATCHSIZE, collate_fn=data_collator
# )

test_dataloader = DataLoader(
    tokenized_data["test"], shuffle=False, batch_size=BATCHSIZE, collate_fn=data_collator
)


# In[20]:


# for batch in train_dataloader:
#     break
# {k: v.shape for k, v in batch.items()}


# ### RoBERTa_model定義

# In[21]:


# # Setting up the device for GPU usage
# from torch import cuda
# device = 'cuda' if cuda.is_available() else 'cpu'


# In[22]:


RoBERTa_model = AutoModelForSequenceClassification.from_pretrained('../module/roberta-base-finetuned-chinanews-chinese')
# for param in RoBERTa_model.parameters():
#     param.requires_grad = False

class RoBERTaClass(torch.nn.Module):
    def __init__(self):  # 建造layer積木
        super(RoBERTaClass, self).__init__()
        self.l1 = RoBERTa_model.base_model
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)  # dense layer類別數量:2
        
    def forward(self, input_ids, attention_mask, token_type_ids):  # 組裝積木
        pooler_output = self.l1(input_ids, attention_mask, token_type_ids)[1]
        output_2 = self.l2(pooler_output)
        output = self.l3(output_2)
        return output
    
model = RoBERTaClass()
model.load_state_dict(torch.load('./save_test_v2.pth', map_location=torch.device('cpu'))) 
model = torch.nn.DataParallel(model)
model.to(device)


# In[23]:


# Loss function定義

LEARNING_RATE = 3e-05

def loss_fn(output, labels):
    return torch.nn.CrossEntropyLoss()(output, labels)

optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)


# In[24]:


train_losses = []

# def train(epoch):
#     model.train() #將 model 設為 training mode
#     total_loss = 0
    
#     for data in tqdm(train_dataloader):
#         input_ids = data['input_ids'].to(device, dtype = torch.long)
#         attention_mask = data['attention_mask'].to(device, dtype = torch.long)
#         token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#         labels = data['labels'].to(device, dtype = torch.long)
        
#         outputs = model(input_ids, attention_mask, token_type_ids)
#         loss = loss_fn(outputs, labels)
#         total_loss += loss.item()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

    
#     train_loss = total_loss/len(train_dataloader)
#     train_losses.append(train_loss)
#     print(f'Epoch:{epoch+1}, Trianing Loss:{total_loss}')
    
    # return train_loss 比較


# In[25]:


eval_losses = []
eval_accu = []
y_pred = []
y_true = []

def evaluation(eval_data):
    model.eval() #將 model 設為 evaluation mode
    total_loss = 0
    total = 0
    correct = 0
    
    
    print('Evaluating...')
    with torch.no_grad():
        for data in tqdm(eval_data):
            input_ids = data['input_ids'].to(device, dtype = torch.long)
            attention_mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.long)

            # print(data)
            # print(len(input_ids))
            # print(token_type_ids)
            outputs = model(input_ids, attention_mask, token_type_ids)
            _, predict = outputs.max(1)
            total += labels.size(0)
            correct += predict.eq(labels).sum().item()
            # loss = loss_fn(outputs, labels)
            # total_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predict.cpu().numpy())
            
    # accu = 100.*correct/total
    # eval_loss = total_loss/len(eval_data)
    # eval_losses.append(eval_loss)
    # eval_accu.append(accu)
    # print(f'Evaluation Loss:{total_loss}, Accuracy:{accu:.3f}%')
    
    
    return y_pred


# ### 訓練及預測

# In[26]:


# 預測

y_pred = []
y_true = []
y_pred=evaluation(test_dataloader)


# In[27]:


# 整理結果
my_predict = df_test_all.join(pd.DataFrame(y_pred))
my_predict.rename(columns={0 : 'drop'},inplace=True)


# In[41]:


# drop之內容
df_drop = my_predict.loc[my_predict['drop']==1]#.loc[:,'title':'date']

# 紀錄過濾掉的資料
drop_final_save_path_file = './../data/no_use_data/'
drop_final_save_path = os.path.join(drop_final_save_path_file,'data_'+date+'.csv')
df_drop.to_csv(drop_final_save_path,index=False)

df_drop


# In[42]:


my_predict.loc[my_predict['drop']==1]


# In[43]:


# 留下非drop之資料並輸出
result = my_predict.loc[my_predict['drop']==0]
result.drop(columns='drop',inplace=True)


# # 列出crawel_type                                   
# crawel_type = [f.split("_")[0] for f in result["filename"]]
# crawel_type

# result['crawel_type'] = crawel_type
# result.drop(columns='filename',inplace=True)


# In[44]:


final_save_path_file = './../data/'
final_save_path = os.path.join(final_save_path_file,'final_'+date+'.csv')
result.to_csv(final_save_path,index=False)


# In[45]:


print('------------------save ok------------------------')
print(f'Save path:{final_save_path}')
print(f'Original data : {len(my_predict)}, Keep data: {len(result)}, Drop data:{len(df_drop)}')
print('-------------------------------------------------')


# In[ ]:





# In[46]:


#If you want to convert all *.ipynb files from current directory to python script, you can run the command like this:
#!pip install ipython
#!pip install nbconvert
#!jupyter nbconvert --to script *.ipynb(轉換成 *.py)

# 只要執行一次, 更新*.py,即可remark, 
# 若要用 docker, 要將 *.py中的 pip install transformer 註解, docker有安裝
#!jupyter nbconvert --to script drop_model.ipynb


# In[ ]:




