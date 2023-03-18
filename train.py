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

from model import NBoW


def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    return {'tokens': tokens}


def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}


def collate(batch, pad_index):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {
        'ids': batch_ids,
        'label': batch_label
    }
    return batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(dataloader, model, criterion, optimizer, device):

    model.train()
    epoch_losses = []
    epoch_accs = []

    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        ids = batch['ids'].to(device)
        label = batch['label'].to(device)
        prediction = model(ids)
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
            label = batch['label'].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

    return epoch_losses, epoch_accs


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


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

    train_data = train_data.with_format(type='torch', columns=['ids', 'label'])
    valid_data = valid_data.with_format(type='torch', columns=['ids', 'label'])

    batch_size = 512
    collate = functools.partial(collate, pad_index=pad_index)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate)

    vocab_size = len(vocab)
    embedding_dim = 300
    output_dim = len(train_data.unique('label'))

    model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    vectors = torchtext.vocab.FastText()
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 10
    best_valid_loss = float('inf')

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    for epoch in range(n_epochs):
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)

        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        valid_losses.extend(valid_loss)
        valid_accs.extend(valid_acc)

        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_valid_loss = np.mean(valid_loss)
        epoch_valid_acc = np.mean(valid_acc)

        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), config['MODEL']['path'])

        print(f'epoch: {epoch + 1}')
        print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
        print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_losses, label='train loss')
    ax.plot(valid_losses, label='valid loss')
    plt.legend()
    ax.set_xlabel('updates')
    ax.set_ylabel('loss')
    plt.savefig(f"{config['OUTPUT']['figures']}/valid_loss_{date.today()}.png")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_accs, label='train accuracy')
    ax.plot(valid_accs, label='valid accuracy')
    plt.legend()
    ax.set_xlabel('updates')
    ax.set_ylabel('accuracy')
    plt.savefig(f"{config['OUTPUT']['figures']}/valid_accuracy_{date.today()}.png")
