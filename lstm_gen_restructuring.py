import os
import numpy as np
import tensorflow as tf
from music21 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adamax

# Ignore warnings and set seed
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


def load_midi_files(filepath):
    """Load MIDI files"""
    return [converter.parse(filepath + i) for i in os.listdir(filepath) if i.endswith(".mid")]


def extract_notes(files):
    """Extract notes from MIDI files"""
    notes = []
    for file in files:
        for element in instrument.partitionByInstrument(file).parts[0].recurse():
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))
    return notes


def preprocess_data(notes, sequence_length=40):
    """Preprocess data"""
    # Create unique symbols and mapping
    symb = sorted(list(set(notes)))
    mapping = dict((c, i) for i, c in enumerate(symb))

    # Generate sequences
    sequences = []
    next_notes = []
    for i in range(0, len(notes) - sequence_length):
        sequences.append([mapping[n] for n in notes[i:i + sequence_length]])
        next_notes.append(mapping[notes[i + sequence_length]])

    # Reshape and normalize data
    X = np.reshape(sequences, (len(sequences), sequence_length, 1)) / float(len(symb))
    y = tf.keras.utils.to_categorical(next_notes)

    return X, y, mapping, len(symb)


def create_model(input_shape, output_shape):
    """Create LSTM model"""
    model = Sequential([
        LSTM(512, input_shape=input_shape, return_sequences=True),
        Dropout(0.1),
        LSTM(256),
        Dense(256),
        Dropout(0.1),
        Dense(output_shape, activation='softmax')
    ])
    opt = Adamax(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model


def train_model(model, X_train, y_train, epochs=3000, batch_size=256):
    """Train the model"""
    return model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)


def generate_melody(model, seed, mapping, num_notes=100, sequence_length=40):
    """Generate melody"""
    reverse_mapping = dict((i, c) for c, i in mapping.items())
    generated_notes = []

    for _ in range(num_notes):
        prediction = model.predict(seed.reshape(1, sequence_length, 1), verbose=0)[0]
        index = np.argmax(prediction)
        generated_notes.append(reverse_mapping[index])
        seed = np.append(seed[1:], [[index / float(len(mapping))]], axis=0)

    return generated_notes


def save_music_to_midi(melody, filename="generated_melody.mid"):
    """Save generated melody to MIDI file"""
    music_stream = stream.Stream()
    for element in melody:
        if '.' in element:  # It's a chord
            chord_notes = element.split('.')
            chord_obj = chord.Chord(chord_notes)
            music_stream.append(chord_obj)
        else:  # It's a note
            n = note.Note(element)
            music_stream.append(n)
    music_stream.write('midi', fp=filename)


def main():
    print("Load and preprocess data")
    filepath = "classical_music_midi/chopin/"
    midi_files = load_midi_files(filepath)
    notes = extract_notes(midi_files)
    X, y, mapping, num_symbols = preprocess_data(notes)

    print("Split training and seed data")
    X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Create and train model")
    model = create_model((X.shape[1], X.shape[2]), y.shape[1])
    history = train_model(model, X_train, y_train)

    model.save('music_lstm_model.h5')
    print("Model saved")

    # Visualize results
    plt.figure(figsize=(15, 4))
    plt.plot(history.history['loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    # plt.show()
    plt.savefig("loss_graph.png")
    plt.close()

    # Generate melody
    seed = X_seed[np.random.randint(0, len(X_seed) - 1)]
    generated_melody = generate_melody(model, seed, mapping)

    save_music_to_midi(generated_melody)

    # Convert generated melody to MIDI (this part is omitted for simplification)
    print("Generated melody length:", len(generated_melody))


if __name__ == "__main__":
    main()
