from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from mymodule.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import progressbar
import numpy as np
import argparse
import random
import h5py
import os

bs = 32

print('... Loading Images')
paths = list(paths.list_images('Flower_17/jpg'))
random.shuffle(paths)

labels = [path.split('/')[-2] for path in paths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print('... Loading Network')
model = VGG16(weights = 'imagenet', include_top = False)

if not os.path.exists('Flower_17/transfer_learning_VGG16'):
    os.mkdir('Flower_17/transfer_learning_VGG16')

dataset = HDF5DatasetWriter(dims = (len(paths), 512*7*7), outPath = 'Flower_17/transfer_learning_VGG16/features.hdf5',
                           datakey = 'features', bufSize = 1000)

dataset.storeClassLabels(le.classes_)

widgets = ['Extracting Features: ', progressbar.Percentage(), ' ' , progressbar.Bar(), 
           ' ', progressbar.ETA(), '\n']
pbar = progressbar.ProgressBar(maxval = len(paths), widgets = widgets)
pbar = pbar.start()

for i in np.arange(0, len(paths), bs):
    batchPaths = paths[i:i+bs]
    batchLabels = labels[i:i+bs]
    batchImages = []
    
    for (j, imgPath) in enumerate(batchPaths):
        image = load_img(imgPath, target_size = (224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        
        batchImages.append(image)
        
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size = 32)
    
    features = features.reshape(features.shape[0], 512*7*7)
    
    dataset.add(features, batchLabels)
    pbar.update(i)
    
dataset.close()
pbar.finish()

"""        
db = h5py.File('Flower_17/transfer_learning_VGG16/features.hdf5')
list(db.keys())

db["features"].shape
db["labels"].shape
db["label_names"].shape
"""







        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


