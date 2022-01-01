from datasets import *
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW, DataCollatorWithPadding, get_scheduler
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import logging
from log_control import *

tokenizer = None
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_ensemble(filename):
  df_ens = pd.read_csv('/content/drive/My Drive/project-shared/Final/output/'+filename+'.csv')
  df_ens['ens_is_humor']=df_ens.filter(regex='^pred_', axis = 1).filter(regex='is_humor$', axis = 1).mode(axis=1)[0]
  df_ens['ens_humor_rating']=df_ens.filter(regex='^pred_', axis = 1).filter(regex='humor_rating$', axis = 1).mean(axis=1)
  df_ens['ens_humor_controversy']=df_ens.filter(regex='^pred_', axis = 1).filter(regex='humor_controversy$', axis = 1).mode(axis=1)[0]
  df_ens['ens_offense_rating']=df_ens.filter(regex='^pred_', axis = 1).filter(regex='offense_rating$', axis = 1).mean(axis=1)
  df_ens.to_csv('/content/drive/My Drive/project-shared/Final/output/'+filename+'_wEns.csv')

  results = {}

  is_humor_f1 = load_metric('f1')
  is_humor_acc = load_metric('accuracy')
  is_humor_f1.add_batch(predictions = df_ens.ens_is_humor.values, references=df_ens.is_humor.values)
  is_humor_acc.add_batch(predictions = df_ens.ens_is_humor.values, references=df_ens.is_humor.values)
  results['is_humor']=(is_humor_f1.compute(), is_humor_acc.compute())

  humor_controversy_f1 = load_metric('f1')
  humor_controversy_acc = load_metric('accuracy')
  humor_controversy_f1.add_batch(predictions = df_ens.ens_humor_controversy.values, references=df_ens.humor_controversy.values)
  humor_controversy_acc.add_batch(predictions = df_ens.ens_humor_controversy.values, references=df_ens.humor_controversy.values)
  results['humor_controversy']=(humor_controversy_f1.compute(), humor_controversy_acc.compute())

  humor_rating_rmse = load_metric('/content/drive/My Drive/project-shared/Final/code/rmse.py')
  humor_rating_rmse.add_batch(predictions = df_ens.ens_humor_rating.values, references=df_ens.humor_rating.values)
  results['humor_rating']=humor_rating_rmse.compute()

  offense_rating_rmse = load_metric('/content/drive/My Drive/project-shared/Final/code/rmse.py')
  offense_rating_rmse.add_batch(predictions = df_ens.ens_offense_rating.values, references=df_ens.offense_rating.values)
  results['offense_rating']=offense_rating_rmse.compute()

  

  return results


def get_split_dfs(src_fname):
  df = pd.read_csv(src_fname).sample(frac=1, random_state = 24).set_index('id')
  df.humor_rating.fillna(0, inplace=True)
  df = df.astype({'is_humor': 'int32'})
  df.humor_controversy.fillna(0, inplace=True)
  df = df.astype({'humor_controversy': 'int32'})
  # df.humor_controversy.astype('int32')
  df_train = df[:int(0.8 * len(df))].copy()
  df_val = df[int(0.8 * len(df)):int(0.9 * len(df))].copy()
  df_test = df[int(0.9 * len(df)):len(df)].copy()

  return df_train, df_val, df_test

def tokenize_function(example):
    return tokenizer(example['text'],truncation=True)

def get_tokenized_datasets(raw_datasets, model_tokenizer, task):
  global tokenizer
  tokenizer = model_tokenizer
  
  # print("Raw Datasets")
  # print (raw_datasets)
  
  col_names = ['is_humor', 'humor_rating', 'humor_controversy', 'offense_rating', 'id']
  col_names.remove(task)
  raw_datasets = raw_datasets.remove_columns(col_names).rename_column(task,'labels')
  
  # print("Raw Datasets after removing columns")
  # print (raw_datasets['test']['text'])
  # print(raw_datasets['test']['labels'])
  
  tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
  
  tokenized_datasets = tokenized_datasets.remove_columns(['text'])

  # print("Tokenized Datasets")
  # print(tokenized_datasets)
  
  tokenized_datasets.set_format("torch")
  return tokenized_datasets
