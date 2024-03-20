# python 3.11
"""
Usage : 
pip install -r requirements.txt

python -m main.py `mode` `test_file` `details`

params:
  mode: str
    The mode to be used, either "dataset" to load dataset on your computer, "train" to train the model and record it under data/my_model.h5, or "test" to test the model previously recorded.
  test_file: Optional[str]
    If mentionned, used to precise the file to test the model, else random file
    eg: mini_speech_commands/yes/0ab3b47d_nohash_0.wav
  details: Optional[bool] = True
    If mentionned, used to precise if plot must appears to help understand the script
"""

import os 
import pathlib
import random
import sys

import tensorflow as tf 
import numpy as np 
import seaborn as sns 
from IPython import display 
from matplotlib import pyplot as plt 
from sklearn.metrics import classification_report

from sixteen_bit_conversion import convert

DATA_DIR = "data"
MINI_SPEECH_DIR = DATA_DIR + "/mini_speech_commands"
RECORD_MODEL_PATH = 'my_model.h5'

def download_dataset():
    # downloading dataset and record it under data folder
    tf.keras.utils.get_file( 
    'mini_speech_commands.zip', 
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip", 
    extract=True, 
    cache_dir='.', cache_subdir=DATA_DIR)



try:
    show_plots = sys.argv[3]
except IndexError:
    show_plots = True

def split():
    # split dataset in train & test sets
    training_set, validation_set = tf.keras.utils.audio_dataset_from_directory( 
        directory=f'{os.getcwd()}/{MINI_SPEECH_DIR}', 
        batch_size=16, 
        validation_split=0.2, 
        output_sequence_length=16000, 
        seed=0, 
        subset='both') 
    return training_set, validation_set
  
# Extracting audio labels 
training_set, validation_set = split()
label_names = np.array(training_set.class_names) 
print("label names:", label_names)

# drop extra axis from audio file
def squeeze(audio, labels): 
  audio = tf.squeeze(audio, axis=-1) 
  return audio, labels 
  
# Applying the function on the dataset obtained from previous step 
training_set = training_set.map(squeeze, tf.data.AUTOTUNE) 
validation_set = validation_set.map(squeeze, tf.data.AUTOTUNE)

# Visualize the waveform 
audio, label = next(iter(training_set)) 
display.display(display.Audio(audio[0], rate=16000))


# Plot the waveform to visualize
def plot_wave(waveform, label): 
    plt.figure(figsize=(10, 3)) 
    plt.title(label) 
    plt.plot(waveform) 
    plt.xlim([0, 16000]) 
    plt.ylim([-1, 1]) 
    plt.xlabel('Time') 
    plt.ylabel('Amplitude') 
    plt.grid(True) 
  
# Convert waveform to spectrogram 
def get_spectrogram(waveform): 
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128) 
    spectrogram = tf.abs(spectrogram) 
    return spectrogram[..., tf.newaxis] 
  
# Plot the spectrogram 
def plot_spectrogram(spectrogram, label): 
    spectrogram = np.squeeze(spectrogram, axis=-1) 
    log_spec = np.log(spectrogram.T + np.finfo(float).eps) 
    plt.figure(figsize=(10, 3)) 
    plt.title(label) 
    plt.imshow(log_spec, aspect='auto', origin='lower') 
    plt.colorbar(format='%+2.0f dB') 
    plt.xlabel('Time') 
    plt.ylabel('Frequency') 
    plt.show()

if show_plots:
    # Plotting the waveform and the spectrogram of a random sample 
    audio, label = next(iter(training_set)) 
    # Plot the wave with its label name 
    plot_wave(audio[0], label_names[label[0]]) 
    # Plot the spectrogram with its label name 
    plot_spectrogram(get_spectrogram(audio[0]), label_names[label[0]])


# Creating spectrogram dataset from waveform or audio data 
def get_spectrogram_dataset(dataset): 
    dataset = dataset.map( 
        lambda x, y: (get_spectrogram(x), y), 
        num_parallel_calls=tf.data.AUTOTUNE) 
    return dataset 
  
# Applying the function on the audio dataset 
train_set = get_spectrogram_dataset(training_set) 
validation_set = get_spectrogram_dataset(validation_set) 
  
# Dividing validation set into two equal val and test set 
val_set = validation_set.take(validation_set.cardinality() // 2) 
test_set = validation_set.skip(validation_set.cardinality() // 2)



# Defining the model 
def get_model(input_shape, num_labels): 
    model = tf.keras.Sequential([ 
        tf.keras.layers.Input(shape=input_shape), 
        # Resizing the input to a square image of size 64 x 64 and normalizing it 
        tf.keras.layers.Resizing(64, 64), 
        tf.keras.layers.Normalization(), 
          
        # Convolution layers followed by MaxPooling layer 
        tf.keras.layers.Conv2D(64, 3, activation='relu'), 
        tf.keras.layers.Conv2D(128, 3, activation='relu'), 
        tf.keras.layers.MaxPooling2D(), 
        tf.keras.layers.Dropout(0.5), 
        tf.keras.layers.Flatten(), 
          
        # Dense layer 
        tf.keras.layers.Dense(256, activation='relu'), 
        tf.keras.layers.Dropout(0.5), 
          
        # Softmax layer to get the label prediction 
        tf.keras.layers.Dense(num_labels, activation='softmax') 
    ]) 
    # Printing model summary 
    model.summary() 
    return model 
  
# Getting input shape from the sample audio and number of classes 
input_shape = next(iter(train_set))[0][0].shape 
print("Input shape:", input_shape) 
num_labels = len(label_names) 

def retrieve_model():
    if sys.argv[1] == "train":
        return get_model(input_shape, num_labels)
    return tf.keras.models.load_model(RECORD_MODEL_PATH)

model = retrieve_model()

def train():
    # Creating a model 

    # train
    model.compile( 
        optimizer="adam", 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy'], 
    ) 
    
    EPOCHS = 10
    history = model.fit( 
        train_set, 
        validation_data=val_set, 
        epochs=EPOCHS, 
    )

    def show_result():
    # Plotting the history 
        metrics = history.history 
        plt.figure(figsize=(10, 5)) 
        
        # Plotting training and validation loss 
        plt.subplot(1, 2, 1) 
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss']) 
        plt.legend(['loss', 'val_loss']) 
        plt.xlabel('Epoch') 
        plt.ylabel('Loss') 
        
        # Plotting training and validation accuracy 
        plt.subplot(1, 2, 2) 
        plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy']) 
        plt.legend(['accuracy', 'val_accuracy']) 
        plt.xlabel('Epoch') 
        plt.ylabel('Accuracy')

        plt.show()

    if show_plots:
        show_result()
    model.save(RECORD_MODEL_PATH)


# validate the model behaviour
    
# Confusion matrix 
y_pred = np.argmax(model.predict(test_set), axis=1) 
y_true = np.concatenate([y for x, y in test_set], axis=0) 
cm = tf.math.confusion_matrix(y_true, y_pred) 
  
# Plotting the confusion matrix 
# plt.figure(figsize=(10, 8)) 
# sns.heatmap(cm, annot=True, fmt='g') 
# plt.xlabel('Predicted') 
# plt.ylabel('Actual') 
# plt.show()


# if show_plots:
#     report = classification_report(y_true, y_pred) 
#     print(report)


def test_model():
    # testing with one sample
    try:
        path = f"{os.getcwd()}/data/{sys.argv[2]}"
        random_label = sys.argv[2].split("/")[1]
    except IndexError:
        # get one of minispeech subfolder
        random_label = random.choice(os.listdir(f"{os.getcwd()}/{MINI_SPEECH_DIR}"))
        # get a random file
        random_file = random.choice(os.listdir(f"{os.getcwd()}/{MINI_SPEECH_DIR}/{random_label}"))
        path = f"{os.getcwd()}/{MINI_SPEECH_DIR}/{random_label}/{random_file}"
    
    print("path:", path, "\n\nrandom_label:", random_label, "\n_____\n")

    # TODO: don't forget to add the test to determine if the sample needs to be converted to 16bits
    convert(path)

    Input = tf.io.read_file(str(path)) 
    x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=16000,) 
    audio, _ = squeeze(x, 'yes') 
    
    waveform = audio 
    display.display(display.Audio(waveform, rate=16000)) 
    
    x = get_spectrogram(audio) 
    x = tf.expand_dims(x, axis=0) 
    plot_wave(audio, random_label) 
    plot_spectrogram(x, random_label)
    
    prediction = model(x) 
    plt.bar(label_names, tf.nn.softmax(prediction[0])) 
    plt.title('Prediction : '+label_names[np.argmax(prediction, axis=1).item()]) 
    plt.show()


if __name__ == "__main__":
    if sys.argv[1] == "dataset":
        download_dataset()

    if sys.argv[1] == "train":
        train()
        test_model()

    if sys.argv[1] == "test":
        test_model()