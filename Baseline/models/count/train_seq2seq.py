
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BartForConditionalGeneration, BartTokenizer
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import  tokenizers
from datasets import load_dataset
import os
import argparse
import json
from torch import nn
import time
from torch.utils.data import random_split
from losses import *
import glob
import logging
from transformers import BartConfig
from torch.utils.tensorboard import SummaryWriter
import copy
from dataset import get_APPDIA_train_and_val_loaders, get_paradetox_train_and_val_loaders

# os.environ["http_proxy"] = "http://127.0.0.1:27999"
# os.environ["https_proxy"] = "http://127.0.0.1:27999"
# Configs
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Train LM')
parser.add_argument('--contrastive_loss', action='store_true')

parser.add_argument('--unlikelihood', action='store_true')

parser.add_argument('--add_negatives', action='store_true')
parser.add_argument('--mapping', action='store_true')
parser.add_argument('--model_name', type=str, default='gpt2')
parser.add_argument('--lm_name', default="/media/data/1/yx/data/model_cache/gpt2-large", type=str)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--alpha', default=0.3, type=float)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--min_delta', default=0.0001, type=float)
parser.add_argument('--patience', default=2, type=int)

parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--gradient_accumulation_steps', default=4, type=int)

parser.add_argument('--dont_save', action='store_true')

parser.add_argument('--save_folder', default="/media/data/3/toxic_model_cache/count-gpt2-large", type=str)


parser.add_argument('--dataset_name', default="paradetox", type=str)

parser.add_argument('--logging_steps', type=int, default=10)

parser.add_argument('--save_steps', type=int, default=3500)

parser.add_argument('--tensorboard_path', type=str, default='/media/data/3/toxic_model_cache/count-gpt2-large/tensorboard')


args = parser.parse_args()
min_delta = args.min_delta
alpha = args.alpha
patience = args.patience
num_epochs = args.num_epochs
mapping = args.mapping
lr = args.lr
lm_name = args.lm_name
# contrastive_loss = args.contrastive_loss
contrastive_loss = True
dataset_name = args.dataset_name

save_folder = args.save_folder

unlikelihood = args.unlikelihood
tensorboard_path = args.tensorboard_path

if dataset_name == 'paradetox':
   
    train_dataloader, eval_dataloader = get_paradetox_train_and_val_loaders(batch_size=args.batch_size)
    print('Paradetox dataset!' ,len(train_dataloader), len(eval_dataloader))
elif dataset_name == 'appdia':
    
    train_dataloader, eval_dataloader = get_APPDIA_train_and_val_loaders(batch_size=args.batch_size)
    print('APPDIA dataset!' ,len(train_dataloader) ,len(eval_dataloader))
else:
    assert False, 'Wrong dataset name!'






if mapping:
    # config_loaded = BartConfig.from_pretrained(lm_name)
    # model = Mapping(config_loaded)

    model = BartForConditionalGeneration.from_pretrained(lm_name)


    # model = Mapping(lm_name,768,[192],768) # add numbers to param
else:
    model = GPT2LMHeadModel.from_pretrained(lm_name)


tokenizer = GPT2Tokenizer.from_pretrained(lm_name)



if mapping:
    optimizer = AdamW(model.model.encoder.layers[-1].parameters(), lr=lr)
else:
    optimizer = AdamW(model.parameters(), lr=lr)



num_training_steps = num_epochs * len(train_dataloader) // args.gradient_accumulation_steps
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print('Device is', device)
model.to(device)


progress_bar = tqdm(total=num_training_steps, desc="Count Training")


model.train()





# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
def combined_contrastive_loss(model ,x ,y):
    return compute_langauge_modeling_loss(model ,x ,y) + alpha * compute_contrastive_loss5(model ,x ,y)


def combined_unlikelihood_loss(model ,x ,y):
    return compute_langauge_modeling_loss(model ,x ,y) + alpha * unlikelihood_loss(model ,x ,y)

def combined_contrastive_loss_with_negatives(model ,x ,y):
    return compute_langauge_modeling_loss(model ,x ,y) + alpha * compute_contrastive_loss8(model ,x ,y)


loss_function = None
if args.contrastive_loss:
    if not args.add_negatives:
        loss_function = combined_contrastive_loss
    else:
        loss_function = combined_contrastive_loss_with_negatives

elif args.unlikelihood:
    loss_function = combined_unlikelihood_loss
else:
    loss_function = compute_langauge_modeling_loss


# EXP_NAME = str(int(time.time()))
# EXP_NAME = 'EXP1'
#
# path_prefix = './saved_models/'
#
#
# if len(save_folder)>0:
#     path_prefix = os.path.join(path_prefix,save_folder)
#
#
# path = os.path.join(path_prefix,EXP_NAME)
path = save_folder

if not args.dont_save:
    if not os.path.exists(path):
        os.makedirs(path)


    with open(os.path.join(path ,'exp_config.json'), "w") as file:
        my_dict = {'alpha' :alpha ,'num_epochs' :num_epochs ,'model_name' :args.model_name ,'patience' :patience
                   ,'min_delta' :min_delta ,'mapping' :mapping,
        'lm_name' :lm_name ,"add_negatives" :args.add_negatives ,'save_folder' :1
                   ,'dataset_name' :dataset_name ,'contrastive_loss' :contrastive_loss ,'unlikelihood' :unlikelihood
       }
        json.dump(my_dict, file)



def save_mapping_checkpint(path ,model ,epoch):

    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.mlp.state_dict(), os.path.join(path ,'mlp.pth'))
    with open(os.path.join(path ,'info.json'), "w") as file:
        my_dict = {'epoch' :epoch}
        json.dump(my_dict, file)

def save_transformers_checkpint(path ,model ,epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save_pretrained(path, from_pt=True) 
    with open(os.path.join(path ,'info.json'), "w") as file:
        my_dict = {'epoch' :epoch}
        json.dump(my_dict, file)


if mapping:
    # save_checkpint =save_mapping_checkpint
    save_checkpint = save_transformers_checkpint
else:
    save_checkpint = save_transformers_checkpint

def get_loss(model ,batch ,loss_function):
    x = tokenizer.batch_encode_plus(batch['en_toxic_comment'], padding=True, truncation=True, return_tensors='pt').to \
        (device)
    y = tokenizer.batch_encode_plus(batch['en_neutral_comment'], padding=True, truncation=True, return_tensors='pt').to \
        (device)
    cat_x = copy.deepcopy(x)
    cat_all = copy.deepcopy(y)
    end_id_batch = [tokenizer.eos_token_id for _ in range(x['input_ids'].shape[0])]
    end_mask_batch = [1 for _ in range(x['input_ids'].shape[0])]
    cat_x['input_ids'] = torch.cat([x['input_ids'], torch.tensor(end_id_batch).unsqueeze(1).to(device)], dim=1)
    cat_x['attention_mask'] = torch.cat([x['attention_mask'], torch.tensor(end_mask_batch).unsqueeze(1).to(device)], dim=1)
    cat_all['input_ids'] = torch.cat([cat_x['input_ids'], y['input_ids']], dim=1)
    cat_all['attention_mask'] = torch.cat([cat_x['attention_mask'], y['attention_mask']], dim=1)

    loss = loss_function(model, cat_all, cat_all)
    return loss


tb_writer = SummaryWriter(tensorboard_path)
global_step = 0
tr_loss, logging_loss = 0.0, 0.0
for epoch in range(num_epochs):

    for batch in train_dataloader:
        # print(batch)
        # batch = {k: v.to(device) for k, v in batch.items()}
        # x = tokenizer.batch_encode_plus(batch['en_toxic_comment'], padding=True, truncation=True, return_tensors='pt').to(device)
        # x = torch.tensor(tokenizer.encode('input: '+batch['en_toxic_comment']+' output:')).unsqueeze(0).to(device)
        # y = tokenizer.batch_encode_plus(batch['en_neutral_comment'], padding=True, truncation=True, return_tensors='pt').to(device)
        # print(batch['en_toxic_comment'])
        # print(x['input_ids'].shape, y['input_ids'].shape, x['attention_mask'].shape)
        # print(x)
        # print(y)

        # print(target_score.shape)

        # loss = loss_function(model,x,y)

        # loss = compute_langauge_modeling_loss(model,x,y) + alpha * compute_contrastive_loss7(model,x,y)

        # loss = compute_langauge_modeling_loss(model,x,y) + alpha * compute_contrastive_loss5(model,x,y)
        loss = get_loss(model ,batch ,loss_function)
        tr_loss += loss.item()
        # loss = compute_contrastive_loss6(model,x,y)

        loss.backward()
        global_step += 1

        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


        if global_step % args.logging_steps == 0:
            tb_writer.add_scalar("lr", lr_scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
            logging_loss = tr_loss

        if global_step % args.save_steps == 0:
            checkpoint_prefix = "checkpoint"
            save_folder = os.path.join(args.save_folder, "{}-{}".format(checkpoint_prefix, global_step))
            os.makedirs(save_folder, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(save_folder)
            tokenizer.save_pretrained(save_folder)

            torch.save(args, os.path.join(save_folder, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", save_folder)

            torch.save(optimizer.state_dict(), os.path.join(save_folder, "optimizer.pt"))
            torch.save(lr_scheduler.state_dict(), os.path.join(save_folder, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", save_folder)
