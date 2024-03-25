import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from matplotlib import pyplot as plt 

# from recording_helper import record_audio, terminate
from test_continuous_record import Recorder
from tf_helper import preprocess_audiobuffer
import pathlib

import input_sender

# !! Modify this in the correct order
train_set = pathlib.Path("data/mini_speech_commands")
commands = np.array(tf.io.gfile.listdir(str(train_set)))
print("Commands:", commands)
loaded_model = models.load_model("my_model.h5")


def predict_mic():
    audio = record_audio()
    
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)

    if not any(float(pred) > 0.8 for pred in prediction[0]):
        print("no satisfying result", prediction[0])
    else:
        label_pred = np.argmax(prediction, axis=1)
        command = commands[label_pred[0]]
        print("Predicted label:", command)
        return command

if __name__ == "__main__":
    input_sender.focus("Pokemon")
    last_input = None
    recorder = Recorder()
    while True:
        command = recorder.listen()
        last_input = input_sender.KeyPress(command, last_input)
            
        # if command == "stop":
        #     terminate()
        #     break