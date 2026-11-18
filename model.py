import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional

CSV_PATH = r"C:\Users\fhs57017\chat\pokemon_prepared.csv"

SAMPLES_PER_CATEGORY = 3000
MAX_SEQUENCE_LENGTH = 20
MAX_VOCAB_SIZE = 5000



df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["message", "category"])
df["message"] = df["message"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()

df = df[(df["message"] != "") & (df["category"] != "")]

df = (
    df.groupby("category", group_keys=False)
    .apply(
        lambda group: group.sample(
            n=min(len(group), SAMPLES_PER_CATEGORY),
            random_state=42
        )
    )
    .reset_index(drop=True)
)

print("\nSamples per category:")
print(df["category"].value_counts())
print("\nTotal samples:", len(df))

texts = []

for row in df.itertuples(index=False):

    message = row.message
    category = row.category

    line = f"CATEGORY {category} CHAT {message} <END>"

    texts.append(line)

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")

tokenizer.fit_on_texts(texts)

total_words = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)

print("\nVocabulary size:", total_words)

sequences_list = []

for line in texts:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):

        n_gram_sequence = token_list[
            max(0, i - MAX_SEQUENCE_LENGTH):i + 1
        ]

        sequences_list.append(n_gram_sequence)


sequences_list = pad_sequences(sequences_list, maxlen=MAX_SEQUENCE_LENGTH + 1, padding="pre",  truncating="pre" )

sequences_list = np.array(sequences_list, dtype=np.int32)

X = sequences_list[:, :-1]
Y = sequences_list[:, -1]

del sequences_list

# Sparse labels (no one-hot encoding)
y = np.asarray(Y, dtype=np.int32)

print("\nTraining samples:", len(X))
print("Input shape:", X.shape)

model = Sequential()

model.add(Embedding(total_words, 128))
model.add(Bidirectional(LSTM(128, dropout=0.2)))
model.add(Dropout(0.3))
model.add(Dense(total_words, activation="softmax"))

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X,
    y,
    epochs=40,
    batch_size=256,
    verbose=1,
    validation_split=0.1
)

# Save model
model.save("twitch_bilstm.keras")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save maximum sequence length
np.save("max_sequence_length.npy", MAX_SEQUENCE_LENGTH)

print("\nTraining completed!")
print("Model saved: twitch_bilstm.keras")
print("Tokenizer saved: tokenizer.pkl")
print("Sequence length saved: max_sequence_length.npy")