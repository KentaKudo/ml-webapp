import sys
sys.path.append('..')

from flask import Flask
from flask_script import Manager, Command
import pickle
from datasets import load_datasets
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense
from sklearn.model_selection import train_test_split

app = Flask(__name__)
manager = Manager(app)

class Train(Command):
    def run(self):
        (X_train, y_train), (X_test, y_test) = load_datasets()
        (X_train, y_train), (X_valid, y_valid) = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        
        x = InceptionV3(input_shape=(225,225))
        model = Dense(1, activation='linear')(x)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train,
                            batch_size=32, epochs=5, verbose=2,
                            validation_data=(X_valid, y_valid))
        score = model.evaluate(x_test, y_test, verbose=0)
        model.save_weights('../weights/jeans.hdf5')
        with open('../weights/history.pkl', 'w') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Loss:', score[0])
        print('Accuracy:', score[1])

if __name__ == "__main__":
    manager.run({
      'train': Train()
    })
