import trafilatura
import re
import json
from gensim.summarization import summarize
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# forbidden words most of them are taken from LIAR-PLUS dataset https://github.com/Tariq60/LIAR-PLUS
FORBIDDEN_WORDS = "misleading|truth|inaccurate|evidence|wrong|incorrect|true|fake|absurd|mistake|exaggerates|barely true|barely-true|mostly false|barely|mostly|claim|true|false|half-true|half true|truth|mostly-ture|mostly true|pants-on-fire|pants on fire|pinocchio|pinocchios"

def extract_summary(link, word_count=200):
  download = trafilatura.fetch_url(link)
  if not download:
      return None
  extract = trafilatura.extract(download)
  if not extract:
      return None
  
  cleaned_str = ""
  for sentence in extract.split("."):
    if not (re.search(FORBIDDEN_WORDS, sentence, flags=re.IGNORECASE)):
        cleaned_str = cleaned_str + sentence
 
  summary = summarize(text=cleaned_str, word_count=200)

  return summary




if __name__ == "__main__":
    dataset = load_dataset("datacommons_factcheck", "fctchk_politifact_wapo")
    data = pd.DataFrame.from_dict(dataset["train"])
    summary = {}
    for i in tqdm(range(0, 2785)) : # data.shape[0])):
        id = data.index[i]
        link = data.iloc[i]['review_url']
        summarize_corpus = extract_summary(link, word_count=200)
        if summarize_corpus:
            summary[str(id)] = {'url_review' : link, 'evidence' : summarize_corpus}
            with open('data/evidences.json', 'w') as outfile:
                json.dump(summary, outfile, indent=4)