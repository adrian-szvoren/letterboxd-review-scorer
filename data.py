import math
import os
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer


def clean_text(text: str) -> str:
    stemmer = WordNetLemmatizer()

    document = re.sub(r'\W', ' ', text)
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    document = re.sub(r'^b\s+', '', document)
    document = document.lower()
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    return document


def process_data(data_path: str) -> pd.DataFrame:
    people = ['Dennis+Schwartz', 'James+Berardinelli', 'Scott+Renshaw', 'Steve+Rhodes']
    features = ['id', 'label.3class', 'label.4class', 'rating', 'subj']

    data = []
    for person in people:
        current_path = f'{data_path}/raw/{person}'
        data_entry = {}

        for feature in features:
            temp = []
            with open(f'{current_path}/{feature}.{person}', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    entry = line.replace('\n', '')
                    temp.append(entry)
            f.close()
            data_entry[feature] = temp

        data_entry = pd.DataFrame(data_entry)
        data_entry['person'] = person

        data.append(data_entry)
    data = pd.concat(data)

    data['clean'] = [clean_text(x) for x in data['subj']]

    data['rating_100'] = [round(float(x) * 100) for x in data['rating']]
    data['rating_imdb'] = [math.ceil(float(x) * 10) for x in data['rating']]
    data['rating_lb'] = [math.ceil(float(x) * 10) / 2 for x in data['rating']]
    data.to_csv(f'{data_path}/processed.csv', index=None)
    return data


def load_data(data_path: str) -> pd.DataFrame:
    if os.path.exists(f'{data_path}/processed.csv'):
        data = pd.read_csv(f'{data_path}/processed.csv')
    else:
        data = process_data(data_path)
    return data
