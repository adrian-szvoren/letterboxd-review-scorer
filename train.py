import configparser
import functools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm

from datasets import Dataset
from datetime import date

from model import NBoW, LSTM, CNN


def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    length = len(tokens)
    return {'tokens': tokens, 'length': length}


def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}


def collate(batch, pad_index):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = [i['length'] for i in batch]
    batch_length = torch.stack(batch_length)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {
        'ids': batch_ids,
        'length': batch_length,
        'label': batch_label
    }
    return batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)


def train(dataloader, model, criterion, optimizer, device):

    model.train()
    epoch_losses = []
    epoch_accs = []

    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        ids = batch['ids'].to(device)
        length = batch['length']
        label = batch['label'].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())

    return epoch_losses, epoch_accs


def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            ids = batch['ids'].to(device)
            length = batch['length']
            label = batch['label'].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(10*[loss.item()])
            epoch_accs.append(10*[accuracy.item()])

    return epoch_losses, epoch_accs


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def plot_and_save_metrics(name, metric_dict):
    metric_df = pd.DataFrame(metric_dict).set_index('epoch')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metric_df)
    plt.legend(metric_df.columns)
    ax.set_xlabel('epoch')
    ax.set_ylabel(name)
    plt.savefig(f"{config['OUTPUT']['figures']}/{config['MODEL']['name']}_{date.today()}_{name}.png")

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    input_path = config['INPUT']['path']
    input_X_col = config['INPUT']['X_col']
    input_y_col = config['INPUT']['y_col']

    df = pd.read_csv(input_path, usecols=[input_X_col, input_y_col])
    df = df.rename(columns={input_X_col: 'text', input_y_col: 'label'})
    dataset = Dataset.from_pandas(df)
    tts = dataset.train_test_split(test_size=0.1)
    train_data, valid_data = tts['train'], tts['test']

    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

    max_length = 256
    train_data = train_data.map(tokenize_example, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
    valid_data = valid_data.map(tokenize_example, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

    min_freq = 5
    special_tokens = ['<unk>', '<pad>']
    vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data['tokens'],
        min_freq=min_freq,
        specials=special_tokens
    )

    unk_index = vocab['<unk>']
    pad_index = vocab['<pad>']
    vocab.set_default_index(unk_index)
    torch.save(vocab, config['VOCAB']['path'])

    train_data = train_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
    valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab': vocab})

    train_data = train_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    valid_data = valid_data.with_format(type='torch', columns=['ids', 'label', 'length'])

    batch_size = 64
    collate = functools.partial(collate, pad_index=pad_index)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate)

    vocab_size = len(vocab)
    embedding_dim = 300
    output_dim = len(train_data.unique('label'))
    model_name = config['MODEL']['name']
    if model_name == 'nbow':
        model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)
        print(f'Using the NBoW model with {count_parameters(model):,} trainable parameters')
    elif model_name == 'lstm':
        hidden_dim = 300
        n_layers = 2
        bidirectional = True
        dropout_rate = 0.5
        model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate,
                     pad_index)
        model.apply(initialize_weights)
    if model_name == 'cnn':
        n_filters = 100
        filter_sizes = [3, 5, 7]
        dropout_rate = 0.25
        model = CNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout_rate, pad_index)
        model.apply(initialize_weights)
    else:
        raise Exception('Model unknown... change the model name in config.ini')
    print(f'Using the {model_name} model with {count_parameters(model):,} trainable parameters')

    vectors = torchtext.vocab.FastText()
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 20
    best_valid_loss = float('inf')
    losses = []
    accuracies = []

    for epoch in range(n_epochs):
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)

        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_valid_loss = np.mean(valid_loss)
        epoch_valid_acc = np.mean(valid_acc)

        losses.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'valid_loss': epoch_valid_loss
        })
        accuracies.append({
            'epoch': epoch + 1,
            'train_acc': epoch_train_acc,
            'valid_acc': epoch_valid_acc
        })

        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), config['MODEL']['path'])

        print(f'epoch: {epoch + 1}')
        print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
        print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')

    plot_and_save_metrics('loss', losses)
    plot_and_save_metrics('accuracy', accuracies)
