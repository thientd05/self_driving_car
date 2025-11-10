from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import numpy as np 
import pandas as pd 
import random
import os

class dataset(keras.utils.Sequence):

    def __init__(self, csv_file, batch_size=32, count=None, img_size=None):
        self.batch_size = batch_size
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file)

        y = self.data.iloc[:,3].values
        pos_zero = np.array(np.where(y == 0)).reshape(-1, 1)
        pos_none_zero = np.array(np.where(y != 0)).reshape(-1, 1)
        np.random.shuffle(pos_zero)
        pos_zero = pos_zero[:750]
        pos_combined = np.vstack((pos_zero,pos_none_zero))
        pos_combined = pos_combined.flatten()

        self.data = self.data.iloc[pos_combined].reset_index(drop=True)

        if count is None:
            self.count = self.data.shape[0]
        else:
            self.count = count
        
        if img_size is None:
            size = load_img(self.data.iloc[0,0]).size 
            self.img_size = (size[1], size[0])
        else:
            self.img_size = img_size

    
    def __len__(self):
        return (self.count) // (self.batch_size)

    
    def get_img_size(self):
        return self.img_size

    def mapping(self, x):
        return (2.0 * x - 1.0)
    
    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_data = self.data.iloc[i: min(i + self.batch_size, len(self.data))]

        
        x = np.zeros((len(batch_data),) + (self.img_size) + (3,), dtype='uint8')
        y = np.zeros((len(batch_data),) + (2,))

        for i in range(len(batch_data)):
            img_path = batch_data.iloc[i,0]
            if os.path.exists(img_path):
                imgc = load_img(img_path, target_size=self.img_size)
                imgc = np.array(imgc)
                steering_angle = batch_data.iloc[i,3]
                throttle = self.mapping(batch_data.iloc[i,4])


                if steering_angle != 0:
                    if random.random() > 0.5:
                        imgc = np.fliplr(imgc)
                        steering_angle = - steering_angle
                
                    if random.random() > 0.5:
                        imgc = np.array(imgc) * random.uniform(0.5, 1.5)
                        imgc = np.clip(imgc, 0, 255)

                x[i] = imgc
                y[i,0] = steering_angle
                y[i,1] = throttle

        return x, y
