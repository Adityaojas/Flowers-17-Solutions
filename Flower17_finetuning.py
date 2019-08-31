from mymodule.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from mymodule.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mymodule.dataset.simpledatasetloader import SimpleDatasetLoader
from mymodule.conv.fcheadnet import FCHeadNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.layers import Input # for network surgery
from keras.models import load_model
from keras.models import Model # for network surgery
from imutils import paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if os.path.exists('Flower_17/finetuning_VGG16') == False:
    os.mkdir('Flower_17/finetuning_VGG16')

aug = ImageDataGenerator(rotation_range = 30, height_shift_range = 0.1, width_shift_range = 0.1,
                         shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True,
                         fill_mode = 'nearest')

path = 'Flower_17/jpg'
listPaths = list(paths.list_images(path))
classNames = [x.split('/')[-2] for x in listPaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors = [aap, iap])

(data, labels) = sdl.load(listPaths, verbose = 50)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

data = data.astype('float') / 255.0
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25, random_state = 111)

baseModel = VGG16(weights = 'imagenet', include_top = False, input_tensor=Input(shape=(224, 224, 3)))
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

model = Model(inputs=baseModel.input, outputs=headModel)

"""
for (i, layer) in enumerate(baseModel.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))
"""

for layer in baseModel.layers:
    layer.trainable = False
    
opt = RMSprop(lr = 0.001)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

fname = 'Flower_17/finetuning_VGG16/weights_rmsprop.hdf5'
checkpoint = ModelCheckpoint(fname, monitor = 'val_acc', mode = 'max', save_best_only = True, verbose = 1)
callbacks = [checkpoint]

H = model.fit_generator(aug.flow(X_train, y_train, batch_size = 32),
                    validation_data = (X_test, y_test), epochs = 25,
                    steps_per_epoch = len(X_train) // 32, callbacks = callbacks, verbose=1)

model = load_model('Flower_17/finetuning_VGG16/weights_rmsprop.hdf5')
y_pred = model.predict(X_test, batch_size=32)


print(classification_report(y_test.argmax(axis = 1), y_pred.argmax(axis = 1)))
report = classification_report(y_test.argmax(axis = 1), y_pred.argmax(axis = 1), output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('Flower_17/finetuning_VGG16/classification_report_rmsprop.csv', index = False)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 25), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, 25), H.history['val_loss'], label = 'validation_loss')
plt.plot(np.arange(0, 25), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, 25), H.history['val_acc'], label = 'validation_acc')
plt.title('Loss and Accuracy')
plt.xlabel('Loss/Acc')
plt.ylabel('# epochs')
plt.legend()
plt.show
plt.savefig('Flower_17/finetuning_VGG16/plot_rmsprop.jpg')


for (i, layer) in enumerate(baseModel.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))
    
for layer in baseModel.layers[15:]:
    layer.trainable = True

opt = SGD(lr = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

fname = 'Flower_17/finetuning_VGG16/weights_sgd.hdf5'
checkpoint = ModelCheckpoint(fname, monitor = 'val_acc', mode = 'max', save_best_only = True, verbose = 1)
callbacks = [checkpoint]

H = model.fit_generator(aug.flow(X_train, y_train, batch_size = 32),
                    validation_data = (X_test, y_test), epochs = 100,
                    steps_per_epoch = len(X_train) // 32, callbacks = callbacks, verbose=1)

model = load_model('Flower_17/finetuning_VGG16/weights_sgd.hdf5')
y_pred = model.predict(X_test, batch_size=32)


print(classification_report(y_test.argmax(axis = 1), y_pred.argmax(axis = 1)))

report = classification_report(y_test.argmax(axis = 1), y_pred.argmax(axis = 1), output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('Flower_17/finetuning_VGG16/classification_report_sgd.csv', index = False)

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
plt.savefig('Flower_17/finetuning_VGG16/plot_sgd.jpg')










