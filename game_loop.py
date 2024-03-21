import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from matplotlib import pyplot as plt 

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer
import pathlib

import input_sender

# !! Modify this in the correct order
train_set = pathlib.Path("data/mini_speech_commands")
commands = np.array(tf.io.gfile.listdir(str(train_set)))
print("Commands:", commands)
loaded_model = models.load_model("my_model.h5")
mappin_commands = {
    "up": 0x2C, # W
    "down": 0x1F, # S
    "stop": 0x01, #esc
    "left": 0x10, #Q
    "right": 0x20, #D
    "go": 0x1C, # enter
    "yes": 0x26, # l
    "no": 0x32, # m
}

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
    while True:
        command = predict_mic()

        if command in mappin_commands:
            input_sender.KeyPress(mappin_commands[command])
            
        # if command == "stop":
        #     terminate()
        #     break