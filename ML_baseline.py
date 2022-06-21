import argparse
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
import pandas as pd
import contractions
from nltk.corpus import stopwords
import nltk
import warnings
warnings.filterwarnings('ignore')


nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=True):
    
    # lower case
    text = text.lower()
    
    # expand contractions
    if True:
        text = text.split()
        new_text = []
        for word in text:
            new_text.append(contractions.fix(word))
        text = " ".join(new_text)
    
    # remove unwanted characters
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    # remove stopwords
    if remove_stopwords:
        text = text.split()
        text = [w for w in text if not w in STOP_WORDS]
        text = " ".join(text)

    return text

# to make sklearn handle the dataframe 
class TextTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None, *parg, **kwarg):
        return self

    def transform(self, X):
        return X[self.key]
    
class CatTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Training.')
    parser.add_argument('-f','--features', help='provide type of features to include in training', required=True, choices=['CLAIM', 'CLAIM_META', 'CLAIM_META_EVIDENCE'])
    args = parser.parse_args()

    # read in train and test dataframe and clean text features
    df_train = pd.read_json('data/train_data.json')
    df_train['claim_text'] = list(map(clean_text, df_train['claim_text']))
    df_train['evidence'] = list(map(clean_text, df_train['evidence']))

    df_test = pd.read_json('data/test_data.json')
    df_test['claim_text'] = list(map(clean_text, df_test['claim_text']))
    df_test['evidence'] = list(map(clean_text, df_test['evidence']))

    claim_text = Pipeline([
                    ('transformer', TextTransformer(key='claim_text')),
                    ('vectorizer', TfidfVectorizer(ngram_range=(1,1), analyzer='word', norm='l2'))
                    ])
    evidence = Pipeline([
                    ('transformer', TextTransformer(key='evidence')),
                    ('vectorizer', TfidfVectorizer(ngram_range=(1,1), analyzer='word', norm='l2'))
                    ])

    claim_name = Pipeline([
                    ('transformer', CatTransformer(key='claim_author_name')),
                    ('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))
                    ])
    poli_party = Pipeline([
                    ('transformer', CatTransformer(key='political_party')),
                    ('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))
                    ])

    if args.features == 'CLAIM':
        features = FeatureUnion([('claim_text', claim_text),
                                ])
        features_name = ['claim_text']

    elif args.features == 'CLAIM_META':
        features = FeatureUnion([('claim_text', claim_text),
                                ('claim_author_name', claim_name),
                                ('political_party', poli_party),
                                ])
        features_name = ['claim_text', 'claim_author_name', 'political_party']

    elif args.features == 'CLAIM_META_EVIDENCE':
        features = FeatureUnion([('claim_text', claim_text),
                                ('evidence', evidence),
                                ('claim_author_name', claim_name),
                                ('political_party', poli_party),
                                ])
        features_name = ['claim_text', 'evidence', 'claim_author_name', 'political_party']

    pipe = Pipeline([('features', features),
                    ('classifier', RandomForestClassifier())
                    ])


    param_grid = [
        {'classifier' : [LogisticRegression(class_weight= 'balanced',multi_class='ovr',)],
        'classifier__penalty' : ['l2'],
        'classifier__C' : np.logspace(-4, 4, 10),
        'classifier__solver' : ['liblinear', 'lbfgs']
        },
        {'classifier' : [RandomForestClassifier()],
        # 'classifier__n_estimators' : list(range(60, 220, 10)),
        # 'classifier__max_features' : list(range(6,32,20))
        },
        {'classifier' : [XGBClassifier()],
        # 'classifier__max_depth': list(range(2, 10, 2)),
        # 'classifier__n_estimators': list(range(60, 220, 30)),
        }
    ]

    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1, scoring='f1_macro')

    # fit the model
    best_clf = clf.fit(df_train[features_name], df_train['label'])

    print(best_clf.best_params_)

    best_clf.best_estimator_.fit(df_train[features_name], df_train['label'])
    preds = best_clf.best_estimator_.predict(df_test[features_name])
    # print out metrics
    print(metrics.classification_report(df_test['label'], preds))