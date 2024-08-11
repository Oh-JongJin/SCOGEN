print('Import modules...', end='')
import os
import tensorflow
import numpy as np
import pandas as pd
from collections import Counter
import random
import IPython
from IPython.display import Image, Audio
import music21
from music21 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# %matplotlib inline
import sys
import warnings

print('OK')

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(42)

# Loading the list of chopin's midi files as stream
filepath = "classical-music-midi/chopin/"
# Getting midi files
all_midis = []

for i in os.listdir(filepath):
    if i.endswith(".mid"):
        tr = filepath + i
        midi = converter.parse(tr)
        all_midis.append(midi)


# Helping function
def extract_notes(file):
    notes = []
    pick = None
    for j in file:
        songs = instrument.partitionByInstrument(j)
        for part in songs.parts:
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


# Getting the list of notes as Corpus
Corpus = extract_notes(all_midis)
print("Total notes in all the Chopin midis in the dataset:", len(Corpus))
