
# NLP for FactChecking

Models to detect TrueNews and FakeNews on the DataCommons FactCheck dataset.

## Prepare dataset

Extract evidence or context of each sample news by scrapping the link associated to the news`review_url`. Sentences that contains the label or words associated to the label are discrated. Since the articles are often long, we extract only a summary with Gensim library. The result is save in `data/evidences.json`.

```
python extract_evidence.py
```
Once the evidence is extracted, run the following to prepare the dataset and extract the political affiliation of each `claim_author_name`.

```
python prepare_datasset.py
```

## Models

### Logistic Regression
- cleaning the text
- TF-IDF vectorization for `claim_tex` and `evidence`
- OneHot encoding for meta data (`claim_author_name` and `political_party`)
- training Classifier (GridSearch and cross-validation)
- returns the best results on the test set
By running the following script, the results will be save under `results/result_metrics.json`.
Choose which features to include.
```
python ML_baseline.py -f [CLAIM, CLAIM_META, CLAIM_META_EVIDENCE]
```
F1-Score:
- CLAIM :
    - TrueNews : 0.57
    - FakeNews : 0.75
    - Macro Avg. : 0.69
- CLAIM + META :
    - TrueNews : 0.60
    - FakeNews : 0.79
    - Macro Avg. : 0.73
- CLAIM + META + EVIDENCE :
    - TrueNews : 0.62
    - FakeNews : 0.80
    - Macro Avg. : 0.74
### LSTM
- vocabulary building
- encoding text data
- training LSTM Models
- returns the result on test set
```
python LSTM.py -f [CLAIM, CLAIM_META, CLAIM_META_EVIDENCE]
```
F1-Score:
- CLAIM :
    - TrueNews : 0.42
    - FakeNews : 0.79
    - Macro Avg. : 0.60
- CLAIM + META :
    - TrueNews : 0.51
    - FakeNews : 0.80
    - Macro Avg. : 0.71
- CLAIM + META + EVIDENCE :
    - TrueNews : 0.48
    - FakeNews : 0.79
    - Macro Avg. : 0.69

### RoBERTa Classifier
- concatening the text features
- tokenizer
- training RoBERTa Classifier
- returns the result on test set
```
python roberta-classifier.py -f [CLAIM, CLAIM_META, CLAIM_META_EVIDENCE]
```
F1-Score:
- CLAIM :
    - TrueNews : 0.42
    - FakeNews : 0.79
    - Macro Avg. : 0.60
- CLAIM + META :
    - TrueNews : 0.51
    - FakeNews : 0.80
    - Macro Avg. : 0.71
- CLAIM + META + EVIDENCE :
    - TrueNews : 0.48
    - FakeNews : 0.79
    - Macro Avg. : 0.69

### Future Works

- Docstrings
- Test Units
- Improve `evidence` extraction
- Improve the Hyperparameters tuning of Models