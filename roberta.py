from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer,BertModel, BertForSequenceClassification
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from torch.utils.data import TensorDataset
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATASET
DATASET_COLUMNS = ["target", "text"]
COLUMNS = ['title', 'id', 'category', 'transcript']
TRAIN_SIZE = 0.8

label_dict = {
    'computer science': 0,
    'biology': 1,
    'environmental studies': 2
}

df = pd.read_csv("./data.csv" , names=DATASET_COLUMNS, usecols = [2, 3], header=0)

print("Dataset size:", len(df))
# print(df)

# ax = sns.countplot(x=df['target'])
# ax.set_xticks(range(0,3))
# ax.set_title("Target Class Distribution in Dataset")
# ax.set_xticklabels(["CS","BIO", "ES"])
# ax.set_xlabel("Categories")
# plt.savefig('./figures/TotalDataDist.png')

DATASET_SIZE = len(df)
df['id'] = [i for i in range(1, DATASET_SIZE+1)]
df.set_index('id', inplace=True)

df['target'] = df['target'].map(label_dict)
X_train, X_test, y_train, y_test = train_test_split(df.index.values, df.target.values, test_size=0.20, random_state=42, stratify=df.target.values)

df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_test, 'data_type'] = 'test'

# print(df.head())

# print(df.groupby(['target', 'data_type']).count())

def train_test_data(encodings):
    dataset_train = TensorDataset(encodings['train_encodings']['input_ids'], encodings['train_encodings']['attention_mask'],encodings['labels_train'])
    dataset_test = TensorDataset(encodings['test_encodings']['input_ids'], encodings['test_encodings']['attention_mask'],encodings['labels_test'])                                
    return (dataset_train, dataset_test)

def get_encodings(tokenizer, df):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    encoded_data_test = tokenizer.batch_encode_plus(
        df[df.data_type=='test'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    labels_train = torch.tensor(df[df.data_type=='train'].target.values)
    labels_test = torch.tensor(df[df.data_type=='test'].target.values)

    return {
        'train_encodings' : encoded_data_train,
        'labels_train'    : labels_train,
        'test_encodings'  : encoded_data_test,
        'labels_test'     : labels_test
    }

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'macro')

def accuracy_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')

def plot_cm(preds, labels, name='roberta'):
    # label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    fig = plt.figure(figsize=(6, 5))
    cm = confusion_matrix(labels_flat, preds_flat)
    cm_plot = sns.heatmap(cm, annot=True, cmap='Blues_r')
    cm_plot.set_xlabel('Predicted Values')
    cm_plot.set_ylabel('True Values')
    if name:
        fig.savefig(name+".png")

def report(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return classification_report(labels_flat, preds_flat)

def evaluate(model, dataloader_test):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_test):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_test) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def model_trainer(model, train_data, test_data, optimizer_params=None):
    optimizer = AdamW(
        model.parameters(),
        **optimizer_params
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps = len(train_data)
    )

    model.to(device)

    model.train()
    loss_train_total = 0
    progress_bar = tqdm(train_data)

    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }
        outputs = model(**inputs)
        loss = outputs[0]
        loss_train_total +=loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})     
        
    tqdm.write('='*15)
    #torch.save(model.state_dict(), f'Models/BERT_ft.model')
    loss_train_avg = loss_train_total/len(train_data)
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    test_loss, predictions, true_vals = evaluate(model, test_data)
    test_f1 = f1_score_func(predictions, true_vals)
    test_accuracy = accuracy_score_func(predictions, true_vals)
    plot_cm(predictions, true_vals)
    tqdm.write(f'Test loss: {test_loss}')
    tqdm.write(f'F1 Score (Macro): {test_f1}')
    tqdm.write(f'Accuracy Score: {test_accuracy}')
    tqdm.write(report(predictions, true_vals))



roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_encodings = get_encodings(roberta_tokenizer, df)
roberta_train, roberta_test = train_test_data(roberta_encodings)
roberta_dataloader_train = DataLoader(
    roberta_train,
    batch_size = 4
)

roberta_dataloader_test = DataLoader(
    roberta_test,
    batch_size = 4
)

roberta_optimizer_params = {
    'lr' : 3e-5
}
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels = 3)
model_trainer(roberta_model, roberta_dataloader_train, roberta_dataloader_test, roberta_optimizer_params)

