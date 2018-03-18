import sys
sys.path.append('..')

from flask import Flask
from flask_script import Manager, Command

app = Flask(__name__)
manager = Manager(app)

class Export(Command):
    def run(self):
        self.plot_model()
        self.plot_history()

    def plot_model(self):
        from datasets import num_classes
        from models import InceptionV3
        from keras.utils import plot_model
        model = InceptionV3(num_classes=num_classes)
        plot_model(model, to_file="../models/inception_v3.png")

    def plot_history(self):
        import pickle
        import matplotlib.pyplot as plt
        history = None
        with open("../weights/history.pkl", "rb") as f:
            history = pickle.load(f)
        plt.plot(history['loss'], "o-", label="loss")
        plt.plot(history['val_loss'], "o-", label="val_loss")
        plt.plot(history['acc'], "o-", label="acc")
        plt.plot(history['val_acc'], "o-", label="val_acc")
        plt.title("Trainig Histories")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='center right')
        plt.savefig("../weights/history.png")

if __name__ == "__main__":
    manager.run({
      'export': Export()
    })
