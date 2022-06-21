import os
import argparse
import json
import pandas as pd
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sklearn.utils import class_weight
from torch import nn
from torch.optim import Adam
from sklearn.metrics import classification_report
from tqdm import tqdm


class FakenewsDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer):

        self.labels = [label for label in df['label']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['input_text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class RoBertaClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(RoBertaClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.roberta(input_ids = input_id, attention_mask = mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def train(model, tokenizer, train_data, val_data, class_weights, batch_size, learning_rate, epochs):

    train = FakenewsDataset(train_data, tokenizer)
    val = FakenewsDataset(val_data, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
            model = model.cuda()
            loss = loss.cuda()

    for epoch_num in range(epochs):

            total_right_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = loss(output, train_label)
                total_loss_train += batch_loss.item()
                
                right = (output.argmax(dim=1) == train_label).sum().item()
                total_right_train += right

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_right_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = loss(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    right = (output.argmax(dim=1) == val_label).sum().item()
                    total_right_val += right
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_right_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_right_val / len(val_data): .3f}')

def evaluate(model, tokenizer, test_data):

    test = FakenewsDataset(test_data, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    pred = []

    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              for p in output.argmax(dim=1).cpu().numpy():
                  pred.append(p)
    return pred                

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='ML Training.')
  parser.add_argument('-f','--features', help='provide type of features to include in training', required=True, choices=['CLAIM', 'CLAIM_META', 'CLAIM_META_EVIDENCE'])
  args = parser.parse_args()

  df_train = pd.read_json('/content/train_data.json')
  df_test = pd.read_json('/content/test_data.json')

  class_weights = class_weight.compute_class_weight('balanced', classes=[0,1], y = df_train['label'])
  class_weights = torch.tensor(class_weights, dtype=torch.float)

  if args.features == 'CLAIM':
    df_train['input_text'] = df_train['claim_text']
    df_test['input_text'] = df_test['claim_text']

  elif args.features == 'CLAIM_META':
    df_train['input_text'] = df_train['claim_text'] + ' </s></s> ' + df_train['claim_author_name'] + ' </s></s> ' + df_train['political_party']
    df_test['input_text'] = df_test['claim_text'] + ' </s></s> ' + df_test['claim_author_name'] + ' </s></s> ' + df_test['political_party']

  elif args.features == 'CLAIM_META_EVIDENCE':
    df_train['input_text'] = df_train['claim_text'] + ' </s></s> ' + df_train['claim_author_name'] + ' </s></s> ' + df_train['political_party'] + ' </s></s> ' + df_train['evidence']
    df_test['input_text'] = df_test['claim_text'] + ' </s></s> ' + df_test['claim_author_name'] + ' </s></s> ' + df_test['political_party'] + ' </s></s> ' + df_test['evidence']

  model = RoBertaClassifier()
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

  train(model, tokenizer, df_train, df_test, class_weights, 2, 2e-6, 3)

  predicted = evaluate(model, tokenizer, df_test)

  metric_out = classification_report(df_test['label'], predicted, output_dict=True)
  # export the classification results
  if not os.path.exists('result/result_metric.json'):
        with open('result/result_metric.json', 'w') as fp:
            json.dump(metric_out, fp, indent=4)
  else:
        with open('result/result_metric.json', 'r+') as fp:
            data = json.load(fp)
            data[f"RoBERTa_{args.features}"] = metric_out
            fp.seek(0)
            json.dump(data, fp, indent=4)

