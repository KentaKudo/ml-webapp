import sys
sys.path.append('..')

from flask import Flask
from flask_script import Manager, Command
import pickle
from datasets import load_datasets
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

app = Flask(__name__)
manager = Manager(app)

class Train(Command):
    def run(self):
        (X_train, y_train), (X_test, y_test) = load_datasets()
        (X_train, X_valid, y_train, y_valid) = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        
        inception_v3 = InceptionV3(weights=None, input_shape=(225,225,3))
        x = Dense(1, activation='linear')(inception_v3.outputs[0])
        model = Model(inception_v3.inputs[0], x)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train,
                            batch_size=32, epochs=5, verbose=2,
                            validation_data=(X_valid, y_valid))
        score = model.evaluate(X_test, y_test, verbose=0)

        # save model
        model.save_weights('../weights/jeans.hdf5')
        with open('../weights/history.pkl', 'wb') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../weights/model.json', 'w') as f:
            f.write(model.to_json())

        print('Loss:', score[0])
        print('Accuracy:', score[1])

if __name__ == "__main__":
    manager.run({
      'train': Train()
    })
