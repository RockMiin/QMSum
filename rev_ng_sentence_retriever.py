import pandas as pd
import numpy as np
import random
import os
import json

from tqdm import tqdm
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import transformers
from transformers import AdamW
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model import DPR
import wandb

def get_config():
    parser = argparse.ArgumentParser()


    """basic, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--qmodel', type=str, default= 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    parser.add_argument('--pmodel', type=str, default= 'sentence-transformers/paraphrase-MiniLM-L6-v2')
    parser.add_argument('--turn', type=int, default= 3, help='feature에 포함될 turn의 수')

    """hyperparameter"""
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--gradient_accum', type=int, default=32,
                        help='gradient accumulation (default: 32)')
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--max_length', type=int, default=384)
    
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

def load_feature(data, turn):
    query= []
    feature= []
    label= []

    tmp_query= ''
    tmp_feature= []
    tmp_labels= []
    for i in tqdm(range(0, len(data))):
        tmp_labels.append(data[i]['label'])
        tmp_feature.append(data[i]['sentence'])

        if (i+1) % turn == 0 or ( i < len(data) -1 and (data[i]['query'] != data[i+1]['query'])):
            tmp_query= data[i]['query']

            if 1 in tmp_labels:
                tmp_label= 1
            else: tmp_label= 0

            query.append(tmp_query)
            feature.append(' \n '.join(tmp_feature))
            label.append(tmp_label)

            tmp_query= ''
            tmp_feature= []
            tmp_labels= []
    
    return {'query': query, 'feature':feature, 'label':label}

def make_ng_sample(data_dict, batch_size):
    query= data_dict['query']
    feature= data_dict['feature']
    label= data_dict['label']

    feature_by_query= []
    label_by_query= []
    query_by_query= []

    tmp_feature= []
    tmp_label= []

    for i in range(len(query)):
        tmp_feature.append(feature[i])
        tmp_label.append(label[i])

        if i < len(query)-1 and query[i] != query[i+1]:
            feature_by_query.append(tmp_feature)
            label_by_query.append(tmp_label)
            query_by_query.append(query[i])

            tmp_feature= []
            tmp_label= []
            # print(len(tmp_feature), len(tmp_label))
    print(f'query length: {len(query)}') # 이것보다 축소가 될 것임..! positive 개수만큼..?
    print(len(feature_by_query))
    
    # query: 1094, feature: 1094 * batch_size
    batch= []
    for i in tqdm(range(len(feature_by_query))): # unique한 쿼리의 개수..!
        
        pos= []
        neg= []
        for j in range(len(feature_by_query[i])): # 쿼리 안의 여러 개의 feature 동안 반복문..
            if label_by_query[i][j] == 1:
                pos.append(feature_by_query[i][j])
            else:
                neg.append(feature_by_query[i][j])

        for j in range(len(pos)):
            neg_idxs = np.random.randint(len(neg), size=batch_size -1).tolist()
            batch_feature= [pos[j]]  + [neg[n] for n in neg_idxs]

            batch.append([query_by_query[i], batch_feature])

    return batch      

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
    feature= []

    for i in range(len(data)):
        query.append(data[i][0])
        feature.extend(data[i][1])

    tokenized_query= tokenizer(query, 
                    max_length= max_length,
                    padding= 'max_length',
                    truncation= True,
                    return_tensors= 'pt')

    tokenized_feature= tokenizer(feature, 
                    max_length= max_length,
                    padding= 'max_length',
                    truncation= True,
                    return_tensors= 'pt')

    tokenized_feature['input_ids']= tokenized_feature['input_ids'].view(-1, batch_size, max_length) 
    tokenized_feature['attention_mask']= tokenized_feature['attention_mask'].view(-1, batch_size, max_length) 
    tokenized_feature['token_type_ids']= tokenized_feature['token_type_ids'].view(-1, batch_size, max_length) 

    # print(tokenized_query['input_ids'].shape)
    # print(tokenized_feature['input_ids'].shape)

    dataset= TensorDataset(
        tokenized_query['input_ids'],
        tokenized_query['attention_mask'],
        tokenized_query['token_type_ids'],
        tokenized_feature['input_ids'],
        tokenized_feature['attention_mask'],
        tokenized_feature['token_type_ids']
    )

    return dataset

def train(q_model, p_model, trainset, validset):

    train_loader= DataLoader(trainset, batch_size= batch_size, shuffle= True)
    valid_loader= DataLoader(validset, batch_size= batch_size, shuffle= True)

    optimizer_grouped_parameters = [{"params": p_model.parameters()}, {"params": q_model.parameters()}]
    optimizer = AdamW(optimizer_grouped_parameters, lr= lr, eps= epsilon)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=lr* 0.1)
    
    q_model.zero_grad()
    p_model.zero_grad()

    torch.cuda.empty_cache()

    for epoch in tqdm(range(epochs)):
        train_loss= 0
        train_acc= 0

        q_model.train()
        p_model.train()

        pbar= tqdm(enumerate(train_loader), total= len(train_loader))
        for step, batch in pbar:

            if torch.cuda.is_available():
                batch = tuple(t.to(device) for t in batch)
                q_model.to(device)
                p_model.to(device)
        
            q_outputs = q_model(
                input_ids= batch[0],
                attention_mask= batch[1],
                token_type_ids= batch[2]
                )

            p_outputs = p_model(
                input_ids= batch[3].view(batch[3].shape[0] *batch_size, -1),
                attention_mask= batch[4].view(batch[3].shape[0] *batch_size, -1),
                token_type_ids= batch[5].view(batch[3].shape[0] *batch_size, -1)
                )

            p_outputs = p_outputs.view(batch[0].shape[0], batch_size, -1)
            q_outputs = q_outputs.view(batch[0].shape[0], 1, -1)

            sim_scores= torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2))

            sim_scores = sim_scores.view(batch[0].shape[0], -1) 
            sim_scores= F.log_softmax(sim_scores, dim= 1)

            targets = torch.zeros(batch[0].shape[0]).long().to(device) # batch 내의 query에 대한 정답은 모두 첫번째 idx의 feature이므로

            loss= F.nll_loss(sim_scores, targets)
            loss= loss / gradient_accum

            preds = torch.argmax(sim_scores.cpu(), axis=1)

            train_acc+= torch.sum(preds== targets.cpu())

            train_loss+= loss.item()
            # print(torch.argmax(sim_scores.cpu(), axis=1))
            print(preds)

            loss.backward()

            if ((step + 1) % gradient_accum) == 0: # gradient accumulation
                optimizer.step()
                scheduler.step()

                p_model.zero_grad()
                q_model.zero_grad()
        
        print(f'Epoch : {epoch}, train loss : {train_loss/len(train_loader)}')
        print(f'train acc: {train_acc/len(train_loader.dataset)}')
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

                q_outputs = q_model(
                    input_ids= batch[0],
                    attention_mask= batch[1],
                    token_type_ids= batch[2]
                    )

                p_outputs = p_model(
                    input_ids= batch[3].view(batch[3].shape[0] *batch_size, -1),
                    attention_mask= batch[4].view(batch[3].shape[0] *batch_size, -1),
                    token_type_ids= batch[5].view(batch[3].shape[0] *batch_size, -1)
                    )
                print(p_outputs)
                q_outputs = q_outputs.view(batch[0].shape[0], 1, -1)
                p_outputs = p_outputs.view(batch[0].shape[0], batch_size, -1)
                sim_scores= torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2))

                sim_scores = sim_scores.view(batch[0].shape[0], -1)
                sim_scores= F.log_softmax(sim_scores, dim= 1)

                targets = torch.zeros(batch[0].shape[0]).long().to(device)

                loss = F.nll_loss(sim_scores, targets)

                preds = torch.argmax(sim_scores.cpu(), axis=1)
                # print(sim_scores)
                print(preds)
                valid_acc+= torch.sum(preds== targets.cpu())
                valid_loss+= loss.item()
        
        wandb.log({'train loss': train_loss/len(train_loader), 'train acc': train_acc/len(train_loader.dataset), \
        'val loss': valid_loss/len(valid_loader), 'val acc': valid_acc/len(valid_loader.dataset)})

        print(f'epoch: {epoch}, val loss: {valid_loss/len(valid_loader)}, val acc: {valid_acc/len(valid_loader.dataset)}')

        print('model save!')
        if not os.path.exists('./model/q_model') and not os.path.exists('./model/p_model'):
            os.makedirs('./model/q_model')
            os.makedirs('./model/p_model')
        torch.save(q_model.state_dict(), f'./model/q_model/pytorch_model_{epoch}.pt')
        torch.save(p_model.state_dict(), f'./model/p_model/pytorch_model_{epoch}.pt')


if __name__ == "__main__":
    args= get_config()
    seed_everything(42)

    batch_size= args.batch_size
    epochs= args.epochs
    lr= args.lr
    epsilon= args.epsilon # optimizer epsilon
    gradient_accum= args.gradient_accum
    max_length= args.max_length
    turn= args.turn

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data load
    train_data= load_data('./data/dpr_train.jsonl')
    valid_data= load_data('./data/dpr_val.jsonl')
    
    train_dict= load_feature(train_data, turn)
    valid_dict= load_feature(valid_data, turn)
    print(f'train data length: {len(train_dict["feature"])}')
    print(f'valid data length: {len(valid_dict["feature"])}')

    train_batch= make_ng_sample(train_dict, batch_size)
    valid_batch= make_ng_sample(valid_dict, batch_size)


    q_model_name= 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    p_model_name= 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    # q_model_name= 'bert-base-uncased'
    # p_model_name= 'bert-base-uncased'
    print('load model !')
    q_model, p_model, tokenizer= load_model(q_model_name, p_model_name)
    # q_model.load_state_dict(torch.load('/home/hadoop/rocky/tmp2/model/q_model/pytorch_model_0.pt'))
    # p_model.load_state_dict(torch.load('/home/hadoop/rocky/tmp2/model/p_model/pytorch_model_0.pt'))

    print('make dataset !')
    trainset= load_dataset(train_batch, tokenizer)
    validset= load_dataset(valid_batch, tokenizer)

    run = wandb.init(
                project="qmsum",
                name='sbert-org-sche',
                group="retriever",
            )

    train(q_model, p_model, trainset, validset)
    # run.finish()

    torch.save(q_model.state_dict(), './model/q_model/pytorch_model.pt')
    torch.save(p_model.state_dict(), './model/p_model/pytorch_model.pt')