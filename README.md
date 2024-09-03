# Music generation with neural network

# MIDI File Processing and Music Generation using TensorFlow

This project focuses on processing MIDI files and generating new musical sequences using machine learning models. The workflow includes MIDI data extraction, analysis, and the training of a sequence model to predict and generate new notes.

Table of Contents
Installation
Usage
Dataset
MIDI File Processing
Model Training
Generating New Music
Results
Contributing
License
Installation
To run this project, you'll need to install the following dependencies:

Fluidsynth - A software synthesizer for generating audio from MIDI files.
Pretty MIDI - A Python library for handling MIDI data.
TensorFlow - An open-source platform for machine learning.
Additional Libraries - numpy, pandas, seaborn, matplotlib, IPython, etc.

Usage
To use this project:

Clone the repository or download the project files.
Mount your Google Drive to access or store the data.
Download the MAESTRO dataset if not already available.
Run the provided code to process the MIDI files and generate new music.
Dataset
The dataset used in this project is the MAESTRO v2.0.0 dataset, which contains approximately 1,200 MIDI files.

MIDI File Processing
The project processes MIDI files to extract musical features like pitch, step, and duration. The steps include:

Loading MIDI Files - Using the pretty_midi library to load and inspect MIDI files.
Note Extraction - Extracting notes and their attributes like pitch, start time, end time, and duration.
Visualization - Plotting the notes in a piano roll format and analyzing their distributions.

Model Training
A neural network model is trained to predict the next note based on the sequence of previous notes. The model consists of LSTM layers and custom loss functions to handle pitch, step, and duration predictions.

Training parameters include:

Sequence Length: 25
Batch Size: 64
Learning Rate: 0.005
Number of Epochs: 50
Model summary and training:

python
Copy code
model = tf.keras.Model(inputs, outputs)
model.compile(loss=loss, optimizer=optimizer)
history = model.fit(train_ds, epochs=epochs, callbacks=callbacks)
Generating New Music
The trained model is used to generate new musical sequences by predicting notes one by one. The generated notes are then converted back into a MIDI file format for playback.


Results
The project successfully generates new MIDI files based on the trained model. The generated sequences are stored in output.mid and can be played back using a MIDI player.

Example Output
output.mid - Generated MIDI file
Plots of the piano roll and note distributions.
