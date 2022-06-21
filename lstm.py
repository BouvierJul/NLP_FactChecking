import os
import json
import torchtext
from torchtext.legacy.data import Field, Dataset, Example, BucketIterator
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.optim import Adam
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse

CLAIM = 0
CLAIM_META = 0
CLAIM_META_EVIDENCE = 0

# class to read pandas Dataframe
class DataFrameDataset(torchtext.legacy.data.Dataset):
    def __init__(self, df: pd.DataFrame, fields: list):
        super(DataFrameDataset, self).__init__(
            [Example.fromlist(list(r), fields) for i, r in df.iterrows()], 
            fields
        )

# vanilla LSTM with no attention
class LSTM(nn.Module):

    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm_long = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=4,
                            batch_first=True,
                            bidirectional=True)
        self.lstm_short = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)
        if CLAIM_META:
          self.fc = nn.Linear(6*dimension, 1)
        elif CLAIM_META_EVIDENCE:
          self.fc = nn.Linear(8*dimension, 1)

    def forward(self, claim, claim_len, author, author_len, party, party_len, evidence, evidence_len):

        # claim branch
        claim_emb = self.embedding(claim)
        packed_input = pack_padded_sequence(claim_emb, claim_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm_long(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), claim_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        claim_fea = self.drop(out_reduced)

        if CLAIM:
          fea = self.fc(claim_fea)
          fea = torch.squeeze(fea, 1)
          out = torch.sigmoid(fea)

        elif CLAIM_META:
          # author branch
          author_emb = self.embedding(author)
          packed_input = pack_padded_sequence(author_emb, author_len, batch_first=True, enforce_sorted=False)
          packed_output, _ = self.lstm_short(packed_input)
          output, _ = pad_packed_sequence(packed_output, batch_first=True)

          out_forward = output[range(len(output)), author_len - 1, :self.dimension]
          out_reverse = output[:, 0, self.dimension:]
          out_reduced = torch.cat((out_forward, out_reverse), 1)
          author_fea = self.drop(out_reduced)

          # party branch
          party_emb = self.embedding(party)
          packed_input = pack_padded_sequence(party_emb, party_len, batch_first=True, enforce_sorted=False)
          packed_output, _ = self.lstm_short(packed_input)
          output, _ = pad_packed_sequence(packed_output, batch_first=True)

          out_forward = output[range(len(output)), party_len - 1, :self.dimension]
          out_reverse = output[:, 0, self.dimension:]
          out_reduced = torch.cat((out_forward, out_reverse), 1)
          party_fea = self.drop(out_reduced)

          fea = torch.cat((claim_fea, author_fea, party_fea),1)
          fea = self.fc(fea)
          fea = torch.squeeze(fea, 1)
          out = torch.sigmoid(fea)

        elif CLAIM_META_EVIDENCE:

          # author branch
          author_emb = self.embedding(author)
          packed_input = pack_padded_sequence(author_emb, author_len, batch_first=True, enforce_sorted=False)
          packed_output, _ = self.lstm_short(packed_input)
          output, _ = pad_packed_sequence(packed_output, batch_first=True)

          out_forward = output[range(len(output)), author_len - 1, :self.dimension]
          out_reverse = output[:, 0, self.dimension:]
          out_reduced = torch.cat((out_forward, out_reverse), 1)
          author_fea = self.drop(out_reduced)

          # party branch
          party_emb = self.embedding(party)
          packed_input = pack_padded_sequence(party_emb, party_len, batch_first=True, enforce_sorted=False)
          packed_output, _ = self.lstm_short(packed_input)
          output, _ = pad_packed_sequence(packed_output, batch_first=True)

          out_forward = output[range(len(output)), party_len - 1, :self.dimension]
          out_reverse = output[:, 0, self.dimension:]
          out_reduced = torch.cat((out_forward, out_reverse), 1)
          party_fea = self.drop(out_reduced)

          # evidence branch
          evi_emb = self.embedding(evidence)
          packed_input = pack_padded_sequence(evi_emb, evidence_len, batch_first=True, enforce_sorted=False)
          packed_output, _ = self.lstm_long(packed_input)
          output, _ = pad_packed_sequence(packed_output, batch_first=True)

          out_forward = output[range(len(output)), evidence_len - 1, :self.dimension]
          out_reverse = output[:, 0, self.dimension:]
          out_reduced = torch.cat((out_forward, out_reverse), 1)
          evi_fea = self.drop(out_reduced)
          
          fea = torch.cat((claim_fea, author_fea, party_fea, evi_fea),1)
          fea = self.fc(fea)
          fea = torch.squeeze(fea, 1)
          out = torch.sigmoid(fea)

        
        return out

def train(model, train_data, val_data, learning_rate, epochs, threshold=0.5):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    loss = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
            model = model.cuda()
            loss = loss.cuda()

    for epoch_num in range(epochs):

            total_right_train = 0
            total_loss_train = 0

            for batch in tqdm(train_data):

                labels = batch.label.to(device)
                claim_text = batch.claim_text[0].to(device)
                claim_text_len = batch.claim_text[1]
                claim_author = batch.claim_author_name[0].to(device)
                claim_author_len = batch.claim_author_name[1]
                poli_party = batch.political_party[0].to(device)
                poli_party_len = batch.political_party[1]
                evidence = batch.evidence[0].to(device)
                evidence_len = batch.evidence[1]
                output = model(claim_text, claim_text_len, claim_author, claim_author_len, poli_party, poli_party_len, evidence, evidence_len)
                batch_loss = loss(output, labels)
                total_loss_train += batch_loss.item()
                # print((output > threshold).int())
                # print(labels.int())
                right = ((output > threshold).int() == labels.int()).sum().item()
                total_right_train += right

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_right_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for batch in val_data:

                    labels = batch.label.to(device)
                    claim_text = batch.claim_text[0].to(device)
                    claim_text_len = batch.claim_text[1]
                    claim_author = batch.claim_author_name[0].to(device)
                    claim_author_len = batch.claim_author_name[1]
                    poli_party = batch.political_party[0].to(device)
                    poli_party_len = batch.political_party[1]
                    evidence = batch.evidence[0].to(device)
                    evidence_len = batch.evidence[1]
                    output = model(claim_text, claim_text_len, claim_author, claim_author_len, poli_party, poli_party_len, evidence, evidence_len)
                    batch_loss = loss(output, labels)
                    total_loss_val += batch_loss.item()
                    # print((output > threshold).int())
                    # print(labels)
                    right = ((output > threshold).int() == labels.int()).sum().item()
                    total_right_val += right
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data.dataset): .3f} | Train Accuracy: {total_right_train / len(train_data.dataset): .3f} | Val Loss: {total_loss_val / len(val_data.dataset): .3f} | Val Accuracy: {total_right_val / len(val_data.dataset): .3f}')

def evaluate(model, test_loader, threshold=0.5):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:           
            labels = batch.label.to(device)
            claim_text = batch.claim_text[0].to(device)
            claim_text_len = batch.claim_text[1]
            claim_author = batch.claim_author_name[0].to(device)
            claim_author_len = batch.claim_author_name[1]
            poli_party = batch.political_party[0].to(device)
            poli_party_len = batch.political_party[1]
            evidence = batch.evidence[0].to(device)
            evidence_len = batch.evidence[1]
            output = model(claim_text, claim_text_len, claim_author, claim_author_len, poli_party, poli_party_len, evidence, evidence_len)


            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

    return y_pred
    # print('Classification Report:')
    # print(classification_report(y_true, y_pred, labels=[0,1], digits=4))

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='ML Training.')
  parser.add_argument('-f','--features', help='provide type of features to include in training', required=True, choices=['CLAIM', 'CLAIM_META', 'CLAIM_META_EVIDENCE'])
  args = parser.parse_args()

  if args.features =='CLAIM':
    CLAIM = 1

  if args.features =='CLAIM_META':
    CLAIM_META = 1

  if args.features =='CLAIM_META_EVIDENCE':
    CLAIM_META_EVIDENCE = 1

  df_train = pd.read_json('data/train_data.json')
  df_test = pd.read_json('data/test_data.json')

  df_train = df_train[['claim_text', 'claim_author_name', 'political_party', 'evidence', 'label']]
  df_test = df_test[['claim_text', 'claim_author_name', 'political_party', 'evidence', 'label']]

  label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
  text_field = Field(tokenize='basic_english', lower=True, include_lengths=True, batch_first=True)

  train_dataset = DataFrameDataset(
      df = df_train, 
      fields = (
        ('claim_text', text_field),
        ('claim_author_name', text_field),
        ('political_party', text_field),
        ('evidence', text_field),
        ('label', label_field)
      )
  )
  test_dataset = DataFrameDataset(
      df = df_test, 
      fields = (
          ('claim_text', text_field),
          ('claim_author_name', text_field),
          ('political_party', text_field),
          ('evidence', text_field),
          ('label', label_field)
      )
  )

  text_field.build_vocab(train_dataset, min_freq=3)
  vocab = text_field.vocab

  train_iter, test_iter = BucketIterator.splits(
    datasets=(train_dataset, test_dataset), 
    batch_sizes=(2, 2),
    sort=False
  )
  model = LSTM()
  train(model, train_iter, test_iter, 0.0003, 3)
  predicted = evaluate(model, test_iter, threshold=0.5)

  metric_out = classification_report(df_test['label'], predicted, output_dict=True)
  # export the classification results
  if not os.path.exists('result/result_metric.json'):
        with open('result/result_metric.json', 'w') as fp:
            json.dump(metric_out, fp, indent=4)
  else:
        with open('result/result_metric.json', 'r+') as fp:
            data = json.load(fp)
            data[f"LSTM_{args.features}"] = metric_out
            fp.seek(0)
            json.dump(data, fp, indent=4)
  # print(classification_report(df_test['label'], predicted))