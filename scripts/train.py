import os, sys
sys.path.append('..')

from flask import Flask
from flask_script import Manager, Command
import pickle
from datasets import load_datasets, num_classes
from models import ResNet, image_size
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

app = Flask(__name__)
manager = Manager(app)

epochs = int(os.environ['EPOCHS']) if 'EPOCHS' in os.environ else 5

class Train(Command):
    def run(self):
        (X_train, y_train), (X_test, y_test) = load_datasets()
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        
        model = ResNet(num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath="../weights/category.hdf5", verbose=1, save_best_only=True)
        history = model.fit(X_train, y_train,
                            batch_size=32, epochs=epochs, verbose=2,
                            validation_split=0.2,
                            callbacks=[checkpointer])
        score = model.evaluate(X_test, y_test, verbose=0)

        # save model
        with open('../weights/history.pkl', 'wb') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../models/resnet.json', 'w') as f:
            f.write(model.to_json())

        print('Loss:', score[0])
        print('Accuracy:', score[1])

if __name__ == "__main__":
    manager.run({
      'train': Train()
    })
