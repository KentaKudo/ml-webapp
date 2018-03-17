from flask import Flask
from flask_script import Manager, Command
import pickle

app = Flask(__name__)
manager = Manager(app)

class Train(Command):
  def run(self):
    (x_train, y_train), (x_test, y_test) = self.load_datasets()

  def load_datasets(self):
    with open("../datasets/jeans.pkl") as handler:
      datasets = pickle.load(handler)
    return datasets

if __name__ == "__main__":
  manager.run({
    'train': Train()
  })
