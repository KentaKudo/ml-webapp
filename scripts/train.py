import sys
sys.path.append('..')

from flask import Flask
from flask_script import Manager, Command
import pickle
from datasets import load_datasets
from models import InceptionV3
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

app = Flask(__name__)
manager = Manager(app)

class Train(Command):
    def run(self):
        (X_train, y_train), (X_test, y_test) = load_datasets()
        
        model = InceptionV3()
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath="../weights/jeans.hdf5", verbose=1, save_best_only=True)
        history = model.fit(X_train, y_train,
                            batch_size=32, epochs=5, verbose=2,
                            validation_split=0.2,
                            callbacks=[checkpointer])
        score = model.evaluate(X_test, y_test, verbose=0)

        # save model
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
