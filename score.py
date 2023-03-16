import pickle

from sklearn.pipeline import Pipeline


def load_model(model_path: str) -> Pipeline:
    pipe = pickle.load(open(f'{model_path}/pipe.pickle', 'rb'))
    return pipe

def score(text: str) -> float:
    model_path = 'model'

    pipe = load_model(model_path)

    score = pipe.predict([text])
    return score
