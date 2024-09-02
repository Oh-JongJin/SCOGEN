import os
import numpy as np
import tensorflow as tf
from datetime import datetime

from music21 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Ignore warnings and set seed
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


def load_midi_files(filepath):
    """Load MIDI files"""
    return [converter.parse(filepath + i) for i in os.listdir(filepath) if i.endswith(".mid")]


def extract_notes(files):
    """Extract notes and durations from MIDI files"""
    notes = []
    durations = []
    for file in files:
        for element in instrument.partitionByInstrument(file).parts[0].recurse():
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                durations.append(element.duration.quarterLength)
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))
                durations.append(element.duration.quarterLength)
    return notes, durations


def preprocess_data(notes, durations, sequence_length=40):
    """Preprocess data"""
    # Create unique symbols and mapping for notes and durations
    note_symb = sorted(list(set(notes)))
    dur_symb = sorted(list(set(durations)))
    note_mapping = dict((c, i) for i, c in enumerate(note_symb))
    dur_mapping = dict((c, i) for i, c in enumerate(dur_symb))

    # Generate sequences
    note_sequences = []
    dur_sequences = []
    next_notes = []
    next_durs = []
    for i in range(0, len(notes) - sequence_length):
        note_sequences.append([note_mapping[n] for n in notes[i:i + sequence_length]])
        dur_sequences.append([dur_mapping[d] for d in durations[i:i + sequence_length]])
        next_notes.append(note_mapping[notes[i + sequence_length]])
        next_durs.append(dur_mapping[durations[i + sequence_length]])

    # Reshape and normalize data
    X_notes = np.reshape(note_sequences, (len(note_sequences), sequence_length, 1)) / float(len(note_symb))
    X_durs = np.reshape(dur_sequences, (len(dur_sequences), sequence_length, 1)) / float(len(dur_symb))
    y_notes = tf.keras.utils.to_categorical(next_notes)
    y_durs = tf.keras.utils.to_categorical(next_durs)

    return X_notes, X_durs, y_notes, y_durs, note_mapping, dur_mapping, len(note_symb), len(dur_symb)


# def create_model(input_shape, output_shape):
def create_model(input_shape, not_output_shape, dur_output_shape):
    """Create LSTM model"""
    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(not_output_shape + dur_output_shape, activation='softmax')
    ])
    opt = Adamax(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, epochs=200, batch_size=256):
    """Train the model"""
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    return model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                     callbacks=[reduce_lr, early_stopping])


def generate_melody(model, seed_notes, seed_durs, note_mapping, dur_mapping, num_notes=100, sequence_length=40):
    """Generate melody"""
    reverse_note_mapping = dict((i, c) for c, i in note_mapping.items())
    reverse_dur_mapping = dict((i, c) for c, i in dur_mapping.items())
    generated_notes = []
    generated_durs = []

    for _ in range(num_notes):
        seed = np.concatenate((seed_notes, seed_durs), axis=-1)
        prediction = model.predict(seed.reshape(1, sequence_length, 2), verbose=0)[0]
        note_pred = prediction[:len(note_mapping)]
        dur_pred = prediction[len(note_mapping):]
        note_index = np.argmax(note_pred)
        dur_index = np.argmax(dur_pred)
        generated_notes.append(reverse_note_mapping[note_index])
        generated_durs.append(reverse_dur_mapping[dur_index])
        seed_notes = np.append(seed_notes[1:], [[note_index / float(len(note_mapping))]], axis=0)
        seed_durs = np.append(seed_durs[1:], [[dur_index / float(len(dur_mapping))]], axis=0)

    return generated_notes, generated_durs


def save_music_to_midi(melody, durations, filename="generated_melody.mid"):
    """Save generated melody to MIDI file"""
    music_stream = stream.Stream()
    for note_str, dur in zip(melody, durations):
        if '.' in note_str:  # It's a chord
            chord_notes = note_str.split('.')
            chord_obj = chord.Chord(chord_notes)
            chord_obj.duration.quarterLength = dur
            music_stream.append(chord_obj)
        else:  # It's a note
            n = note.Note(note_str)
            n.duration.quarterLength = dur
            music_stream.append(n)
    music_stream.write('midi', fp=filename)


def main():
    print("Load and preprocess data")
    now = datetime.now().strftime("%Y%m%d%H%M%S")

    composer = 'chopin'
    filepath = f"classical_music_midi/{composer}/"
    midi_files = load_midi_files(filepath)
    notes, durations = extract_notes(midi_files)
    X_notes, X_durs, y_notes, y_durs, note_mapping, dur_mapping, num_note_symbols, num_dur_symbols = (
        preprocess_data(notes, durations))

    print("Split training and seed data")
    X_train_notes, X_seed_notes, y_train_notes, y_seed_notes = train_test_split(X_notes, y_notes, test_size=0.2,
                                                                                random_state=42)
    X_train_durs, X_seed_durs, y_train_durs, y_seed_durs = train_test_split(X_durs, y_durs, test_size=0.2,
                                                                            random_state=42)

    model = create_model((X_notes.shape[1], 2), y_notes.shape[1], y_durs.shape[1])
    X_train = np.concatenate((X_train_notes, X_train_durs), axis=-1)
    y_train = np.concatenate((y_train_notes, y_train_durs), axis=-1)
    history = train_model(model, X_train, y_train)

    model.save(f'music_lstm_model_{composer}_{now}.h5')
    print("Model saved")

    # Visualize results
    plt.figure(figsize=(15, 4))
    plt.plot(history.history['loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"loss_graph_{now}.png")
    plt.close()

    # Generate melody
    seed_index = np.random.randint(0, len(X_seed_notes) - 1)
    seed_notes = X_seed_notes[seed_index]
    seed_durs = X_seed_durs[seed_index]
    generated_notes, generated_durs = generate_melody(model, seed_notes, seed_durs, note_mapping, dur_mapping)

    save_music_to_midi(generated_notes, generated_durs)

    # Convert generated melody to MIDI (this part is omitted for simplification)
    print("Generated melody length:", len(generated_notes))


if __name__ == "__main__":
    main()
