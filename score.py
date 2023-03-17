import pickle

from sklearn.pipeline import Pipeline
from typing import List


def load_model(model_path: str) -> Pipeline:
    pipe = pickle.load(open(f'{model_path}/pipe.pickle', 'rb'))
    return pipe


def score(text: str) -> List[float]:
    model_path = 'model'

    pipe = load_model(model_path)

    score = pipe.predict([text])
    return score
