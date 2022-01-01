from datasets import *
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW, DataCollatorWithPadding, get_scheduler
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

is_dev_mode = False

tokenizer = None
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def loss_fn(predictions = None, gTruth = None, crit = 'ce'):
  if crit == 'ce':
    loss = torch.nn.functional.cross_entropy(predictions, gTruth)
  else:
    loss = torch.nn.functional.mse_loss(predictions, gTruth)
  return loss


def train_loop(model, optimizer, train_data, eval_data, epochs):
  print("Training Loop...")
  num_training_steps = epochs * len(train_data)
  every=int(num_training_steps*(100/640))
  if model.task in ['is_humor', 'humor_controversy']:
    best_perf = -1
  else:
    best_perf = 100
  best_model = model
  if is_dev_mode:
    epochs = 1
    num_training_steps = 1
    every = 1
  lr_scheduler = get_scheduler('linear', optimizer = optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
  progress_bar = tqdm(range(num_training_steps))

  
  for epoch in range(epochs):
    iter_count = 0
    print()
    print("Started Epoch %d/%d:" % (epoch+1,epochs))
    for batch in train_data:
      model.train()
      iter_count+=1
      batch = {k: v.to(device) for k, v in batch.items()}
      # if ctr == 1:
      #   for k,v in batch.items():
      #     print (k)
      #     print(v.size())
      # ctr += 1
      
      # print(batch['attention_mask'].size())
      # print(batch['input_ids'].size())
      # print(batch['labels'].size())
      # print(batch['token_type_ids'].size())
      
      output = model(batch)
      if model.task in ['is_humor', 'humor_controversy']:
        loss = loss_fn(output, batch['labels'], 'ce')
      else:
        loss = loss_fn(output.squeeze(1), batch['labels'], 'mse')
      loss.backward()

      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      progress_bar.update(1)
      
      if iter_count%every==0 or iter_count==1 or iter_count==len(train_data):
        perfs, _ = eval_model(model, eval_data)
        print("Iteration %d/%d: | Loss = %0.2f | %s" % (iter_count,len(train_data),loss, perfs))
        for key in perfs[0]:
          if model.task in ['is_humor', 'humor_controversy']:
            if perfs[0][key]> best_perf:
                best_perf = perfs[0][key]
                best_model = model
          else:
            if perfs[0][key]< best_perf:
              best_perf = perfs[0][key]
              best_model = model
      if is_dev_mode:
        break
  print("Best Performance:", best_perf)  
  return best_model

def eval_model(model, data):
  model.eval()
  if model.task == 'is_humor' or model.task=='humor_controversy':
    metrics = [load_metric('accuracy'), load_metric('f1')]
  else:
    metrics = [load_metric('/content/drive/My Drive/project-shared/Final/code/rmse.py')]
  pred = []
  
  for batch in data:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
      output = model(batch)
      
      if model.task in ['is_humor', 'humor_controversy']:
        y_hat = torch.argmax(output, dim=1, keepdim=False)
      else:
        y_hat = output.squeeze(1)
      
      pred+= y_hat.tolist()
      

      for metric in metrics:
        metric.add_batch(predictions=y_hat, references=batch['labels'])
  
  perfs = []
  for metric in metrics:
    perfs.append(metric.compute())

  return perfs, pred


