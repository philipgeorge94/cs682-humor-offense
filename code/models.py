from datasets import *
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW, DataCollatorWithPadding, get_scheduler
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import *
from solver import *

dropout_prob= 0

class base_model(torch.nn.Module):

  '''
  Defining the base model:
  1) for a classification task, we call AutoModelForSequenceClassification with output size = num_labels
  2) For a regression task we call AutoModel and then layer an FCNet with output size = 1
  '''

  def __init__(self,checkpoint = 'bert-base-uncased', setup = 'single', task = 'is_humor', has_decoder=False, fc_dim=256, lstm_dim = 256):
    super(base_model, self).__init__()
    self.encoder_type = checkpoint
    self.setup = setup
    self.task = task
    self.has_decoder = has_decoder
    self.out_size = 2
    self.fc_dim = fc_dim
    if self.task in ['humor_rating', 'offense_rating']: self.out_size=1

    if self.setup == 'single':
      self.encoder = AutoModel.from_pretrained(self.encoder_type, output_hidden_states=True)
      if not self.has_decoder:
        self.fc_net = torch.nn.Sequential(torch.nn.Linear(in_features= 768,out_features= self.fc_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(p = dropout_prob), 
                                          torch.nn.Linear(in_features=self.fc_dim, out_features=self.out_size))
      elif self.has_decoder:
        self.lstm = torch.nn.LSTM(input_size= 768, hidden_size= lstm_dim, dropout= dropout_prob, batch_first=True)
        self.fc_net = torch.nn.Sequential(torch.nn.Linear(in_features= lstm_dim,out_features= self.fc_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(p = dropout_prob), 
                                          torch.nn.Linear(in_features=self.fc_dim, out_features=self.out_size))
        
  
  def forward(self, batch):
    output = None
    if self.encoder_type=='roberta-base':
      embedding = self.encoder(input_ids = batch['input_ids'], attention_mask=batch['attention_mask'])
    else:
      embedding = self.encoder(input_ids = batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids= batch['token_type_ids'])

    if self.setup=='single':
      if not self.has_decoder:
        output = self.fc_net(embedding['last_hidden_state'][:,0,:])

      elif self.has_decoder:
        _,out_lstm = self.lstm(embedding['last_hidden_state'])
        output = self.fc_net(out_lstm[0].squeeze())
           
    
    return output