import sys
sys.path.append('..')

from flask import Flask
from flask_script import Manager, Command, Option

app = Flask(__name__)
manager = Manager(app)

class Model(Command):
    def run(self):
        from datasets import num_classes
        from models import ResNet50
        from keras.utils import plot_model
        model = ResNet50(num_classes=num_classes)
        plot_model(model, to_file="../models/resnet50.png")

class History(Command):

    option_list = (
        Option('--filename', '-f', dest='filename', default='history'),
    )

    def run(self, filename):
        import matplotlib
        matplotlib.use('Agg')
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
        plt.savefig("../weights/"+filename+".png")

if __name__ == "__main__":
    manager.run({
      'model': Model(),
      'history': History(),
    })
