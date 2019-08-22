from mymodule.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mymodule.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from mymodule.dataset.simpledatasetloader import SimpleDatasetLoader
from mymodule.conv.minivggnet import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imutils
import cv2
import os

if not os.path.exists('Flower_17/minivggnet_aug'):
    os.mkdir('Flower_17/minivggnet_aug')

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

aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1, height_shift_range = 0.1,
                         shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True,
                         vertical_flip = True, fill_mode = 'nearest')




opt = SGD(lr = 0.03, decay = 0.03 / 100, momentum = 0.9, nesterov = True)
    
model = MiniVGGNet.build(64, 64, 3, 17)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

fname = 'Flower_17/minivggnet_aug/weights.hdf5'
checkpoint = ModelCheckpoint(fname, monitor = 'val_acc', mode = 'max', save_best_only = True, verbose = 1)
callbacks = [checkpoint]

batch_size = 32
steps_per_epoch = len(X_train) // batch_size

H = model.fit_generator(aug.flow(X_train, y_train, batch_size=batch_size), validation_data = (X_val, y_val),
                        callbacks = callbacks, steps_per_epoch = steps_per_epoch,
                        epochs = 100, verbose = 1)

y_pred = model.predict(X_val, batch_size=32)
print(classification_report(y_val.argmax(axis=1), y_pred.argmax(axis=1)))

report = classification_report(y_val.argmax(axis = 1), y_pred.argmax(axis = 1), output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('Flower_17/minivggnet_aug/classification_report.csv', index = False)

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
plt.savefig('Flower_17/minivggnet_aug/minivggnet_flowers17.jpg')
