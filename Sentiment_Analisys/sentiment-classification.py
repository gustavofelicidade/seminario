
import os
import json
import tensorflow as tf
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import tensorflow as tf

import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras import regularizers



embedding_dim = 100
#  max_length e padding_type
#  são parâmetros que especificam como preenchemos as sequências
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 160000  # especifica o tamanho do  target corpus.
test_portion = .1  # define a proporção dos dados que usaremos como validação

#  são hiperparâmetros para o modelo:
dropout_val = 0.2
nof_units = 64

corpus = []


# ====================================================================
# Vamos encapsular a etapa de criação
# do modelo em uma função.
# Definimos um bastante simples para
# nossa tarefa de classificação
# ====================================================================
#  —>    an embedding layer, followed by regularization and convolution, pooling,
#   and then the RNN layer:

def create_model(dropout_val, nof_units):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim,
                                  input_length=max_length,
                                  weights=[embeddings_matrix],
                                  trainable=False),
        tf.keras.layers.Dropout(dropout_val),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(nof_units),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

#  Colete o conteúdo do corpus sobre o qual iremos treinar:

num_sentences = 0

train_csv = "../dataset/train.csv"
cleaned_csv = "../dataset/training.1600000.processed.noemoticon.csv"
cleaned_csv_1 = "../dataset/training.1600000.processed.noemoticon-1.csv"

with open(cleaned_csv, encoding='cp437') as csvfile:
    #reader = csv.reader(csvfile, delimiter=',')
    reader = csv.reader(csvfile,
                        delimiter=',')
#  encoding='utf-8',

    for row in reader:
        list_item = []
        list_item.append(row[5])
        this_label = row[0]
        if this_label == '0':
            list_item.append(0)

        else:
            list_item.append(1)
        num_sentences = num_sentences + 1
        corpus.append(list_item)
        #print(row)


#  Converter para o formato de frase:

sentences = []
labels = []
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

#  Tokenize as frases:

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index)
sequences = tokenizer.texts_to_sequences(sentences)

#  Normalize os comprimentos das frases com padding
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


#  Divida o conjunto de dados em conjuntos de treinamento e validação:
split = int(test_portion * training_size)
test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

test_sequences = np.array(test_sequences)
training_sequences = np.array(training_sequences)
test_labels = np.array(test_labels)
training_labels = np.array(training_labels)

#  Uma etapa crucial no uso de modelos baseados em RNN
#  para aplicativos NLP é a matriz de embeddings:

embeddings_index = {}
with open('../dataset/glove.6B.100d.txt', encoding='cp437') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector


#  Com todos os preparativos concluídos, podemos montar o modelo:

model = create_model(dropout_val, nof_units)
summary = model.summary()
summary




#  O treinamento é realizado da maneira usual:

num_epochs = 10
history = model.fit(training_sequences,
                    training_labels,
                    epochs=num_epochs,
                    validation_data=(test_sequences,
                                     test_labels),
                    verbose=2)

#  Também podemos avaliar a qualidade do nosso modelo visualmente:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
