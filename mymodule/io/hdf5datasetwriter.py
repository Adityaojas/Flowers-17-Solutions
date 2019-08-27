import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outPath, datakey = 'images', bufSize = 1000):
        if os.path.exists(outPath):
            raise ValueError("The output path already exists"
                             "and cannot be overwritten."
                             "Please delete the ouput path manually before continuing", outPath)
            
        self.database = h5py.File(outPath, 'w')
        self.data = self.database.create_dataset(datakey, shape = dims, dtype = 'float')
        self.labels = self.database.create_dataset('labels', shape = (dims[0],), dtype = 'int')
        self.bufSize = bufSize
        
        self.buffer = {'data':[], 'labels':[]}
        self.id = 0
            
    def flush(self):
        i = self.id + len(self.buffer['data'])
        self.data[self.id:i] = self.buffer['data']
        self.labels[self.id:i] = self.buffer['labels']
        self.id = i
        self.buffer = {'data':[], 'labels':[]}
    
    def add(self, rows, labels):
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)
        
        if len(self.buffer['data']) >= self.bufSize:
            self.flush()
            
    def storeClassLabels(self, classLabels):
        
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.database.create_dataset('label_names', shape = (len(classLabels),), dtype = dt)
        labelSet[:] = classLabels
        
    def close(self):
        if len(self.buffer['data']) > 0:
            self.flush()
        
        self.database.close()
        
    
    
    
