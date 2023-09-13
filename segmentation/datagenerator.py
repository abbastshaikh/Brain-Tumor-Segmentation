#%% Import Packages / Setup

import os
import numpy as np
import pandas as pd
import random
import skimage
from keras.utils import to_categorical, Sequence
import nibabel as nib

import warnings
warnings.filterwarnings(action = 'once')

#%% Generators
def horizontal_flip(image_array):
    return image_array[:, ::-1]
  
def vertical_flip(image_array):
    return image_array[::-1]
  
def blur(image_array):
    kernel = np.array([1.0, 4.0, 1.0])
    image_array = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, image_array)
    image_array = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, image_array)
    return image_array

def noise(image_array):
    return skimage.util.random_noise(image_array, mode='gaussian', seed=None, clip=True)

available_transformations = {
    'horizontal_flip': horizontal_flip,
    'vertical_flip': vertical_flip,
    'blur': blur,
    'noise': noise
}  

class AxialGenerator (Sequence):
    
    # Initialization
    def __init__(self, ids, mount, shuffle=True, batch_size = 31):
        
        self.list_IDs = pd.Series([j + "_" + str(i) for i in range(int(155 / batch_size)) for j in ids])
        self.mount = mount
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
    
    # Steps per epoch
    def __len__(self):
        return len(self.list_IDs)

    # Generates batch of data
    def __getitem__(self, index):

        batch_index = self.indexes[index]
        batch_id = self.list_IDs.iloc[batch_index]

        # Generate data
        X, y = self.__data_generation(batch_id)

        return X, y
    
    # Update indices after each epoch
    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            random.shuffle(self.indexes)

    #Generates data with batch size samples
    def __data_generation(self, batch_id):
        
        path = self.mount + "/" + batch_id[:-2] + "/" 
        batch_num = batch_id[-1:]
        
        num_transformations = random.randint(0, len(available_transformations))
        transformations = random.sample(list(available_transformations), num_transformations)
     
        for root, dirs, files in os.walk(path):
            
            for file in files:
                
                img = nib.load(root + '/' + file)
                data = np.array(img.get_fdata())
                
                data = np.swapaxes(data, 0, 2)
                data = np.swapaxes(data, 1, 2)
                data = (data-np.min(data))/(np.max(data)-np.min(data))
                data = data[int(batch_num) * self.batch_size : (int(batch_num) + 1) * self.batch_size]
                    
                if "seg." not in file:
                    for key in transformations:
                        for i in range(data.shape[0]):
                            data[i] = available_transformations[key](data[i])
                
                if "seg." in file: 
                   for key in transformations:
                       if key != 'blur' and key != 'noise':
                           for i in range(data.shape[0]):
                               data[i] = available_transformations[key](data[i])
            
                if "flair." in file:                        
                    flair = np.expand_dims(data, axis = -1)
                
                if "t1." in file:
                    t1 = np.expand_dims(data, axis = -1)
                
                if "t1ce." in file:
                    t1ce = np.expand_dims(data, axis = -1)
                
                if "t2." in file:
                    t2 = np.expand_dims(data, axis = -1)
                
                if "seg." in file: 
                    labels = [0, 1, 2, 4]
                    data = to_categorical(data, 5)
                    masks = np.take(data, labels, axis = -1)
                                
        return [flair, t1, t1ce, t2], masks
    
class CoronalGenerator (Sequence):
    
    # Initialization
    def __init__(self, ids, mount, shuffle=True, batch_size = 30):
        
        self.list_IDs = pd.Series([j + "_" + str(i) for i in range(240 / batch_size) for j in ids])
        self.mount = mount
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
    
    # Steps per epoch
    def __len__(self):
        return len(self.list_IDs)

    # Generates batch of data
    def __getitem__(self, index):

        batch_index = self.indexes[index]
        batch_id = self.list_IDs.iloc[batch_index]

        # Generate data
        X, y = self.__data_generation(batch_id)

        return X, y
    
    # Update indices after each epoch
    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            random.shuffle(self.indexes)

    #Generates data with batch size samples
    def __data_generation(self, batch_id):
        
        path = self.mount + "/" + batch_id[:-2] + "/" 
        batch_num = batch_id[-1:]
        
        num_transformations = random.randint(0, len(available_transformations))
        transformations = random.sample(list(available_transformations), num_transformations)
     
        for root, dirs, files in os.walk(path):
            
            for file in files:
                
                img = nib.load(root + '/' + file)
                data = np.array(img.get_fdata())
                
                data = data[int(batch_num) * self.batch_size : (int(batch_num) + 1) * self.batch_size]
                
                data = np.append(data, np.zeros((240, 240, 5)), axis = -1)   
                data = np.swapaxes(data, 0, 1)
                data = data.astype(float)
               
                if "seg." not in file:
                    data = (data-np.min(data))/(np.max(data)-np.min(data))
                    
                    for key in transformations:
                        for i in range(data.shape[0]):
                            data[i] = available_transformations[key](data[i])
                
                if "seg." in file: 
                    for key in transformations:
                        if key != 'blur' and key != 'noise':
                            for i in range(data.shape[0]):
                                data[i] = available_transformations[key](data[i])
            
                if "flair." in file:                        
                    flair = np.expand_dims(data, axis = -1)
                
                if "t1." in file:
                    t1 = np.expand_dims(data, axis = -1)
                
                if "t1ce." in file:
                    t1ce = np.expand_dims(data, axis = -1)
                
                if "t2." in file:
                    t2 = np.expand_dims(data, axis = -1)
                
                if "seg." in file: 
                    labels = [0, 1, 2, 4]
                    data = to_categorical(data, 5)
                    masks = np.take(data, labels, axis = -1)
                    
        return [flair, t1, t1ce, t2], masks
    
class SaggitalGenerator (Sequence):
    
    # Initialization
    def __init__(self, ids, mount, shuffle=True, batch_size = 30):
        
        self.list_IDs = pd.Series([j + "_" + str(i) for i in range(240 / batch_size) for j in ids])
        self.mount = mount
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
    
    # Steps per epoch
    def __len__(self):
        return len(self.list_IDs)

    # Generates batch of data
    def __getitem__(self, index):

        batch_index = self.indexes[index]
        batch_id = self.list_IDs.iloc[batch_index]

        # Generate data
        X, y = self.__data_generation(batch_id)

        return X, y
    
    # Update indices after each epoch
    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            random.shuffle(self.indexes)

    #Generates data with batch size samples
    def __data_generation(self, batch_id):
        
        path = self.mount + "/" + batch_id[:-2] + "/" 
        batch_num = batch_id[-1:]
        
        num_transformations = random.randint(0, len(available_transformations))
        transformations = random.sample(list(available_transformations), num_transformations)
     
        for root, dirs, files in os.walk(path):
            
            for file in files:
                
                img = nib.load(root + '/' + file)
                data = np.array(img.get_fdata())
                
                data = data[int(batch_num) * self.batch_size : (int(batch_num) + 1) * self.batch_size]
                
                data = np.append(data, np.zeros((240, 240, 5)), axis = -1)   
                data = data.astype(float)
               
                if "seg." not in file:
                    data = (data-np.min(data))/(np.max(data)-np.min(data))
                    
                    for key in transformations:
                        for i in range(data.shape[0]):
                            data[i] = available_transformations[key](data[i])
                
                if "seg." in file: 
                    for key in transformations:
                        if key != 'blur' and key != 'noise':
                            for i in range(data.shape[0]):
                                data[i] = available_transformations[key](data[i])
            
                if "flair." in file:                        
                    flair = np.expand_dims(data, axis = -1)
                
                if "t1." in file:
                    t1 = np.expand_dims(data, axis = -1)
                
                if "t1ce." in file:
                    t1ce = np.expand_dims(data, axis = -1)
                
                if "t2." in file:
                    t2 = np.expand_dims(data, axis = -1)
                
                if "seg." in file: 
                    labels = [0, 1, 2, 4]
                    data = to_categorical(data, 5)
                    masks = np.take(data, labels, axis = -1)
                    
        return [flair, t1, t1ce, t2], masks
    
    