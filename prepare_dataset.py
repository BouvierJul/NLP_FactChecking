from datasets import load_dataset
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import wikipedia
import json
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Make the label binary, remove_news are news from Flip-O-Meter or are neither true or false
FAKE_NEWS = ['false', 'mostly false', 'pants on fire', 'four pinocchios', 'three pinocchios', 'two pinocchios', 'wrong', 'distorts the facts']
TRUE_NEWS = ['true', 'half true', 'mostly true', 'one pinnochio', 'geppetto checkmark']
REMOVE_NEWS = ['full flop', 'needs context', 'lacks context', 'flip- flop', 'half flip', 'flip flop', 'disputed', 'out of context', 'cherry picks', 'not the whole story']

POLITICAL_PARTY = ['Democratic', 'Republican', 'Independent', 'Libertarian']

def find_politic(query):
    suggestion = wikipedia.search(query)
    if not suggestion:
      return "none"
    i = 0
    while True:
      try:
          URL = 'https://en.wikipedia.org/wiki/' + suggestion[i].replace(" ", "_")
          break
      except wikipedia.PageError:
          try:
            URL = wikipedia.page(suggestion[i]).url
            break
          except wikipedia.PageError:
            i+=1
            if i == len(suggestion):
              break
            pass

    res = requests.get(URL).text
    soup = BeautifulSoup(res,'lxml')
    try:
      table = soup.find('table', class_='infobox vcard')
      links = table.find_all(['th','td'])
      for j, link in enumerate(links):
        if "Political party" in link:
          return links[j+1].a.get('title')
        elif "National affiliation" in link:
          return links[j+1].a.get('title')
      return "none"
    except AttributeError:
      return "none"

if __name__ == "__main__":
    dataset = load_dataset("datacommons_factcheck", "fctchk_politifact_wapo")

    # create dataframe
    df = pd.DataFrame.from_dict(dataset["train"])
    df['review_rating'] = df['review_rating'].str.lower()

    # remove rare review_ratings < 5 appearances
    df_reduced = df.groupby('review_rating').filter(lambda x: len(x) >= 5)

    # make the label binary
    data = df_reduced[~df_reduced['review_rating'].isin(REMOVE_NEWS)]
    data['label'] = np.where(data['review_rating'].isin(TRUE_NEWS), 0, 1)

    # if number of claims is less than 3 by the author, mark the author as 'other'
    nb_claim = data['claim_author_name'].value_counts()
    data['claim_author_name'] = np.where(data['claim_author_name'].isin(nb_claim.index[nb_claim > 3]), data['claim_author_name'], 'other')

    # extract political affiliation
    authors = []
    partys = []
    for i, author in tqdm(enumerate(data['claim_author_name'].unique())):
        authors.append(author)
        partys.append(find_politic(author))

    # insert political affiliation in dataframe
    for j in range(len(authors)):
        data.loc[data['claim_author_name'] == authors[j], "political_party"] = partys[j]
    for pp in POLITICAL_PARTY:
        data.loc[data['political_party'].str.contains(pp), "political_party"] = pp

    # load evidences
    with open('data/evidences.json') as json_file:
        dict_evi = json.load(json_file)

    df_evi = pd.DataFrame.from_dict(dict_evi,orient='index').drop(columns='url_review')
    df_evi.index = df_evi.index.map(int)

    # merge the 2 dataframes
    data_fin = pd.concat([data, df_evi], axis=1).dropna()

    # create json files
    df_train, df_test = train_test_split(data_fin, test_size=0.2, random_state=42)

    df_train.to_json(r'data/train_data.json')
    df_test.to_json(r'data/test_data.json')