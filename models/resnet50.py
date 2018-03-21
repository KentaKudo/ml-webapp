from . import image_size

def ResNet50(num_classes):
    from keras.applications.resnet50 import ResNet50 as Rn
    return Rn(weights=None, input_shape=image_size+(3,), classes=num_classes)
