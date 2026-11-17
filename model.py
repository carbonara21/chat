import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional

df = pd.read_csv("/Users/felipecarbone/PycharmProjects/CHAT/pokemon_prepared.csv")
df = df.dropna(subset=["message", "category"])

texts = []
for _, row in df.iterrows():
    message = str(row["message"]).strip()
    category = str(row["category"]).strip()
    if not message or not category:
        continue

    line = (
        f"CATEGORY {category} "
        f"CHAT {message} "
        f"<END>"
    )

    texts.append(line)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

sequences_list = []

# Split into lines
for line in texts:
    # Convert each line into a sequence of tokens
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        # Create the input context
        n_gram_sequence = token_list[:i + 1]
        sequences_list.append(n_gram_sequence)

# Make everything the same length
max_sequence_length = max(len(seq) for seq in sequences_list)
sequences_list = pad_sequences(sequences_list, maxlen=max_sequence_length, padding='pre')
sequences_list = np.array(sequences_list)

X = sequences_list[:, :-1]
Y = sequences_list[:, -1]
# Onehot encoding
y = tf.keras.utils.to_categorical(Y,num_classes=total_words)

model = Sequential()

model.add(Embedding(total_words, 100))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.1)))

model.add(Dropout(0.3))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=75, verbose=1, validation_split=0.1)

model.save("twitch_bilstm.keras")