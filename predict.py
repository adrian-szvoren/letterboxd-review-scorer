import configparser

import pandas as pd
import torch
import torchtext

from model import NBoW, LSTM


def predict_score(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = [vocab[t] for t in tokens]
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    scores = []
    for x in range(10):
        scores.append({'score': x + 1, 'confidence': probability[x].item()})
    return pd.DataFrame(scores)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = torch.load(config['VOCAB']['path'])
    vocab_size = len(vocab)
    pad_index = vocab['<pad>']
    embedding_dim = 300
    output_dim = 10
    model_name = config['MODEL']['name']
    if model_name == 'nbow':
        model = NBoW(vocab_size, embedding_dim, output_dim, pad_index).to(device)
    elif model_name == 'lstm':
        hidden_dim = 300
        n_layers = 2
        bidirectional = True
        dropout_rate = 0.5
        model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate,
                     pad_index).to(device)
    else:
        raise Exception('Model unknown... change the model name in config.ini')
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

    model.load_state_dict(torch.load(config['MODEL']['path']))

    texts = [
        'terrible',
        'bad',
        'good',
        'great',
        'perfect',
        'incredible',
        'unique'
    ]

    for text in texts:
        scores = predict_score(text, model, tokenizer, vocab, device)
        print(text)
        # print(scores)
        scores['mul'] = scores['score'] * scores['confidence']
        top = scores.iloc[scores['confidence'].idxmax()]
        avg_score = sum(scores['mul'])
        print('\t', top['score'], top['confidence'], avg_score)
