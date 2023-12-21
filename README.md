# Music-Generation-By-Mood

This project leverages deep learning techniques and emotion analysis to generate music based on user input or predefined moods. The model is implemented in a Jupyter Notebook (music_generation.ipynb) and utilizes the Spotify Basic Pitch model for efficient conversion of WAV files to MIDI files with minimal information loss. The dataset used for training and testing the model is available on Kaggle: [Music Classification Dataset.](https://www.kaggle.com/datasets/shanmukh05/music-classification)

## Overview
The model is trained using TensorFlow and Keras, employing LSTM layers for music generation. It incorporates a pre-trained emotion classification model from the Transformers library to map user input text to a predicted mood. The generated mood is then used to create music that aligns with the user's emotions.

## Table of Contents
- Dependencies
- Data Preprocessing
- Music Generation
- Mood Classification
- User Input
- Dependencies

Ensure that you have the following dependencies installed before running the Jupyter Notebook:

- TensorFlow
- Keras
- midiutil
- music21
- librosa
- wave
- python_play
- IPython
- tqdm
- pickle
- glob
- transformers

You can install these dependencies using the following:

#### !pip install tensorflow keras midiutil music21 librosa wave python_play IPython tqdm transformers

## Data Preprocessing
The data preprocessing section in the notebook includes functions to convert audio files (in WAV format) into MIDI files using the Spotify Basic Pitch model. The MIDI files are then organized into a structured dataset for further training.

## Music Generation
The music generation section involves functions for generating music sequences based on a given mood. The model is trained using LSTM layers, and the generated notes are saved in a 'Notes' folder.

To train the model for all moods with respective checkpoints for each mood uncomment the train for each mood loop in def main():

To generate music for a given mood, use the provided function:

    # for mood in ['Aggressive', 'Dramatic', 'Happy', 'Romantic', 'Sad']:
    #     train_network(mood)

## Mood Classification
The mood classification section uses a pre-trained emotion classification model from the Transformers library. It maps user input text to a predicted mood, which is then used for music generation.

## User Input
The notebook provides two options for the user:

- Generate music based on their own text input.
- Generate music based on predefined moods (Aggressive, Dramatic, Happy, Romantic, Sad).

Choose the desired option by running the cells in the Jupyter Notebook. For option 1, enter your thoughts or emotions when prompted. The system will predict the emotion, map it to a mood, and generate music accordingly. For option 2, specify the desired mood when prompted, and the system will generate music based on the selected mood.

## User Interface
We have also designed a simple user interface for the user to either put in there texts or a specified mood to generate and play music for them.
