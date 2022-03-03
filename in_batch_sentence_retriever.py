"""
    단순하게 relevant span과 query를 이용하여 in-batch negative sampling으로 학습
"""

import pandas as pd
import numpy as np
import random
import os
import json
import argparse

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import transformers
from transformers import AdamW
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model import DPR

# import wandb

def get_config():
    parser = argparse.ArgumentParser()

    """basic, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--qmodel', type=str, default= 'facebook/dpr-question_encoder-single-nq-base')
    parser.add_argument('--pmodel', type=str, default= 'facebook/dpr-ctx_encoder-single-nq-base')

    """hyperparameter"""
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training (default: 2)')
    parser.add_argument('--gradient_accum', type=int, default=64,
                        help='gradient accumulation (default: 32)')
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--max_length', type=int, default=512)
    
    args= parser.parse_args()

    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_data(path):
    data= []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))

    return data

def load_model(q_model_name, p_model_name):

    # q_config= AutoConfig.from_pretrained(q_model_name)
    # p_config= AutoConfig.from_pretrained(p_model_name)

    # q_model= AutoModel.from_pretrained(q_model_name, config= q_config)
    # p_model= AutoModel.from_pretrained(p_model_name, config= p_config)

    tokenizer= AutoTokenizer.from_pretrained(p_model_name)
    q_model, p_model= DPR(q_model_name, p_model_name)._load_model()

    return q_model, p_model, tokenizer

def load_dataset(data, tokenizer):

    query= []
    relevant_span= []
    for i in range(len(data)):
        query.append(data[i]['query'])
        relevant_span.append(data[i]['relevant_span'])
    
    tokenized_query= tokenizer(query, 
                    max_length= 512,
                    padding= 'max_length',
                    truncation= True,
                    return_tensors= 'pt')

    tokenized_passage= tokenizer(relevant_span, 
                    max_length= 512,
                    padding= 'max_length',
                    truncation= True,
                    return_tensors= 'pt')

    dataset= TensorDataset(
        tokenized_query['input_ids'],
        tokenized_query['attention_mask'],
        tokenized_query['token_type_ids'],
        tokenized_passage['input_ids'],
        tokenized_passage['attention_mask'],
        tokenized_passage['token_type_ids'],
    )

    return dataset

def train(q_model, p_model, trainset, validset):

    train_loader= DataLoader(trainset, batch_size= args.batch_size, shuffle= False)
    valid_loader= DataLoader(validset, batch_size= args.batch_size, shuffle= False)

    optimizer_grouped_parameters = [{"params": p_model.parameters()}, {"params": q_model.parameters()}]
    optimizer = AdamW(optimizer_grouped_parameters, lr= args.lr, eps= args.epsilon)
    
    q_model.zero_grad()
    p_model.zero_grad()

    torch.cuda.empty_cache()

    for epoch in tqdm(range(args.epochs)):
        train_loss= 0

        q_model.train()
        p_model.train()

        pbar= tqdm(enumerate(train_loader), total= len(train_loader))
        for step, batch in pbar:

            if torch.cuda.is_available():
                batch = tuple(t.to(device) for t in batch)
                q_model.to(device)
                p_model.to(device)
        
            p_outputs = p_model(
                    input_ids= batch[3],
                    attention_mask= batch[4],
                    token_type_ids= batch[5]
                    )
            q_outputs = q_model(
                input_ids= batch[0],
                attention_mask= batch[1],
                token_type_ids= batch[2]
            )

            targets = torch.arange(0, batch[0].shape[0]).long().to(device) # In-batch negative sampling

            sim_scores= torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1)) # (1, hidden dim) * (hidden dim, batch)
            sim_scores= F.log_softmax(sim_scores, dim= 1)

            loss= F.nll_loss(sim_scores, targets)
            loss= loss / args.gradient_accum

            train_loss+= loss.item()
            
            loss.backward()

            if ((step + 1) % args.gradient_accum) == 0:
                optimizer.step()
                
                p_model.zero_grad()
                q_model.zero_grad()
        
        print(f'Epoch : {epoch}, train loss : {train_loss/len(train_loader)}')

        """ valid """
        valid_loss, valid_acc= 0, 0

        p_model.eval()
        q_model.eval()

        with torch.no_grad():
            val_pbar= tqdm(enumerate(valid_loader), total= len(valid_loader))
            for idx, batch in val_pbar:
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                    q_model.to(device)
                    p_model.to(device)
                
                p_outputs = p_model(
                    input_ids= batch[3],
                    attention_mask= batch[4],
                    token_type_ids= batch[5]
                    )
                q_outputs = q_model(
                    input_ids= batch[0],
                    attention_mask= batch[1],
                    token_type_ids= batch[2]
                )

                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                targets = torch.arange(0, batch[0].shape[0]).long().to(device)

                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, targets)

                loss = loss
                preds = torch.argmax(sim_scores.cpu(), axis=1)

                valid_acc+= torch.sum(preds== targets.cpu())
                valid_loss+= loss.item()
        
        # wandb.log({'train loss': train_loss/len(train_loader), 'val loss': valid_loss/len(valid_loader), 'val acc': valid_acc/len(valid_loader.dataset)})

        print(f'epoch: {epoch}, val loss: {valid_loss/len(valid_loader)}, acc: {valid_acc/len(valid_loader.dataset)}')



if __name__ == "__main__":

    args= get_config()
    seed_everything(args.seed)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data= load_data('./data/specific_train.jsonl')
    valid_data= load_data('./data/specific_val.jsonl')

    q_model, p_model, tokenizer= load_model(args.qmodel, args.pmodel)

    trainset= load_dataset(train_data, tokenizer)
    validset= load_dataset(valid_data, tokenizer)

    # run = wandb.init(
    #             project="qmsum",
    #             name='facebook-dpr',
    #             group="retriever",
    #         )

    train(q_model, p_model, trainset, validset)
    # run.finish()

    torch.save(q_model.state_dict(), './model/q_model/pytorch_model.pt')
    torch.save(p_model.state_dict(), './model/p_model/pytorch_model.pt')