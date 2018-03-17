from flask import Flask
from flask_script import Manager, Command
import pickle
from ../datasets import load_datasets

app = Flask(__name__)
manager = Manager(app)

class Train(Command):
  def run(self):
    (X_train, y_train), (X_test, y_test) = load_datasets()
    print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)

if __name__ == "__main__":
  manager.run({
    'train': Train()
  })
