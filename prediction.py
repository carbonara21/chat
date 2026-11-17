import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import threading
import queue

# Load trained model
model = tf.keras.models.load_model("twitch_bilstm.keras")

# Load tokenizer
with open("tokenizer.json", "r") as file:
    tokenizer = tokenizer_from_json(file.read())

max_sequence_length = int(np.load("max_sequence_length.npy"))

# Shared category and input handling
current_category = "battle_win"
category_lock = threading.Lock()
input_queue = queue.Queue()
stop_event = threading.Event()


def generate_text(category, max_words=30, temperature=0.8):

    seed_text = f"CATEGORY {category} CHAT"
    generated_words = []

    for _ in range(max_words):

        token_list = tokenizer.texts_to_sequences(
            [seed_text]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_length - 1,
            padding="pre",
            truncating="pre"
        )

        predictions = model.predict(
            token_list,
            verbose=0
        )[0]

        # Apply temperature
        predictions = np.asarray(predictions).astype("float64")
        predictions = np.log(predictions + 1e-8) / temperature

        probabilities = np.exp(
            predictions - np.max(predictions)
        )

        probabilities /= np.sum(probabilities)

        # Select next word
        predicted_index = np.random.choice(
            len(probabilities),
            p=probabilities
        )

        word = tokenizer.index_word.get(
            predicted_index,
            ""
        )

        if word == "<end>":
            break

        if not word or word == "<OOV>":
            continue

        generated_words.append(word)
        seed_text += " " + word

    return " ".join(generated_words)


# Read category changes in a separate thread
def category_input():

    while not stop_event.is_set():

        try:
            new_category = input(
                "\nEnter new category (or 'exit'): "
            ).strip()

            if new_category.lower() == "exit":
                stop_event.set()
                break

            if new_category:
                input_queue.put(new_category)

        except EOFError:
            stop_event.set()
            break


def main():

    global current_category

    print("\n==============================")
    print("     TWITCH CHAT GENERATOR")
    print("==============================")

    current_category = input(
        "\nEnter starting category: "
    ).strip()

    if not current_category:
        current_category = "battle_win"

    # Start input thread
    input_thread = threading.Thread(
        target=category_input,
        daemon=True
    )
    input_thread.start()

    print("\nGenerating messages every second.")
    print("Enter a new category whenever you want.")
    print("Type 'exit' to stop.\n")

    try:
        while not stop_event.is_set():

            # Check for category changes
            while not input_queue.empty():
                current_category = input_queue.get()
                print(f"\nCategory changed to: {current_category}")

            category = current_category

            message = generate_text(
                category=category,
                max_words=30,
                temperature=0.8
            )

            if message:
                print(f"\n[{category}] {message}")

            stop_event.wait(1)

    except KeyboardInterrupt:
        print("\nStopping generator...")

    finally:
        stop_event.set()
        print("Twitch Chat Generator closed.")


if __name__ == "__main__":
    main()
