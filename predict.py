import configparser

import pandas as pd
import torch
import torchtext
import transformers

from model import NBoW, LSTM, CNN, Transformer


def predict_score(text, model, tokenizer, vocab, device, min_length, pad_index):
    tokens = tokenizer(text)
    ids = [vocab[t] for t in tokens]
    if len(ids) < min_length:
        ids += [pad_index] * (min_length - len(ids))
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    scores = []
    for x in range(10):
        scores.append({'score': x + 1, 'confidence': probability[x].item()})
    return pd.DataFrame(scores)


def load_models(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = torch.load(config['VOCAB']['path'])
    vocab_size = len(vocab)
    pad_index = vocab['<pad>']
    embedding_dim = config['MODEL']['embedding_dim']
    output_dim = config['MODEL']['output_dim']
    model_name = config['MODEL']['name']
    min_length = 1
    if model_name == 'nbow':
        model = NBoW(vocab_size, embedding_dim, output_dim, pad_index).to(device)
    elif model_name == 'lstm':
        hidden_dim = int(config['LSTM']['hidden_dim'])
        n_layers = int(config['LSTM']['n_layers'])
        bidirectional = eval(config['LSTM']['bidirectional'])
        dropout_rate = 0
        model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate,
                     pad_index).to(device)
    elif model_name == 'cnn':
        n_filters = int(config['CNN']['n_filters'])
        filter_sizes = eval(config['CNN']['filter_sizes'])
        min_length = max(filter_sizes)
        dropout_rate = 0
        model = CNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout_rate,
                    pad_index).to(device)
    elif model_name == 'transformer':
        transformer_name = config['TRANSFORMER']['name']
        freeze = True
        transformer = transformers.AutoModel.from_pretrained(transformer_name)
        model = Transformer(transformer, output_dim, freeze).to(device)
    else:
        raise Exception('Model unknown... change the model name in config.ini')
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    model.load_state_dict(torch.load(config['MODEL']['path'], map_location=device))
    return model, tokenizer, vocab, device, min_length, pad_index


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    model, tokenizer, vocab, device, min_length, pad_index = load_models(config)

    texts = [
        'terrible',
        'not bad, terrible',
        'bad',
        'good',
        'great',
        'perfect',
        'incredible',
        'unique'
    ]

    for text in texts:
        scores = predict_score(text, model, tokenizer, vocab, device, min_length, pad_index)
        print(text)
        top = scores.iloc[scores['confidence'].idxmax()]
        print(
            '\t',
            scores['score'][scores['confidence'].idxmax()],
            scores['confidence'][scores['confidence'].idxmax()]
        )
