from tensorflow.python.keras.models import load_model
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
  return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
  return top_k_categorical_accuracy(y_true, y_pred, k=2)




model = load_model("path to your saved model",custom_objects={'top_2_accuracy': top_2_accuracy,'top_3_accuracy': top_3_accuracy})


import cv2
import numpy as np
import glob
finename ='nv.jpg'
img = cv2.imread(finename)
img = cv2.resize(img, (224,224))
a = np.expand_dims(img, 0)
prediction = model.predict(a)



classes = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

classes = {v: k for k, v in classes.items()}

classes[np.argmax(prediction)]
