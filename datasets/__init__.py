import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def load_datasets():
  with open(os.path.dirname(__file__)+"/jeans.pkl", 'rb') as f:
    datasets = pickle.load(f)
    X = [v['x'] for v in datasets]
    y = [v['y'] for v in datasets]
    (X_train, X_test, y_train, y_test) = train_test_split(np.array(X),
                                                          np.array(y),
                                                          test_size=0.2,
                                                          random_state=42)
  return (X_train, y_train), (X_test, y_test)
