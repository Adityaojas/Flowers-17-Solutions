from mymodule.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mymodule.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from mymodule.dataset.simpledatasetloader import SimpleDatasetLoader
from mymodule.conv.minivggnet import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imutils
import cv2
import os

if not os.path.exists('Flower_17/minivggnet_wo_aug'):
    os.mkdir('Flower_17/minivggnet_wo_aug')

path = 'Flower_17/jpg'

image_paths = list(paths.list_images(path))
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()
 
sdl = SimpleDatasetLoader(preprocessors = [aap, iap])

(data,labels) = sdl.load(image_paths, verbose=80)
data = data.astype('float32') / 255.0

lb = LabelBinarizer()
labels  = lb.fit_transform(labels)

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size = 0.2, random_state = 111)

opt = SGD(lr = 0.01)
model = MiniVGGNet.build(64, 64, 3, 17)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

fname = 'Flower_17/minivggnet_wo_aug/weights_adam.hdf5'
checkpoint = ModelCheckpoint(fname, monitor = 'val_acc', mode = 'max', save_best_only = True, verbose = 1)
callbacks = [checkpoint]

H = model.fit(X_train, y_train, validation_data = (X_val, y_val), callbacks = callbacks, batch_size = 32, epochs = 100, verbose = 1)

y_pred = model.predict(X_val, batch_size=32)
print(classification_report(y_val.argmax(axis=1), y_pred.argmax(axis=1)))

report = classification_report(y_val.argmax(axis = 1), y_pred.argmax(axis = 1), output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('Flower_17/minivggnet_wo_aug/classification_report.csv', index = False)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label = 'validation_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label = 'validation_acc')
plt.title('Loss and Accuracy')
plt.xlabel('Loss/Acc')
plt.ylabel('# epochs')
plt.legend()
plt.show
plt.savefig('Flower_17/minivggnet_wo_aug/minivggnet_cifar10.jpg')







