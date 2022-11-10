# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 05:45:33 2022

@author: Raksha Rajendran
"""

from PIL import Image #to read and write images
import numpy as np

#OCR using KNN

#filenames
DATA_DIR = 'data/'
TEST_DIR = 'test/'
TEST_DATA_FILENAME=DATA_DIR+'t10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR+'t10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME =DATA_DIR + 'train-labels.idx1-ubyte'


def read_image(path):
    return np.array(Image.open(path).convert('L'))

def write_image(image,path):
    img = Image.fromarray(np.array(image),'L')
    img.save(path)



#conversion
def bytes_to_int(byte_data):
    return int.from_bytes(byte_data,'big')


#reading labels of traing data
def read_labels(filename,n_max_labels=None):
    labels=[]
    with open(filename,'rb') as f:
        _ = f.read(4)
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = f.read(1)
            labels.append(label)
    return labels
            
         



#reading images of training data 
def read_images(filename,n_max_images=None):
    images=[]
    with open(filename,'rb') as f:
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4))
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        for image_idx in range(n_images):
            image=[]
            for row_idx in range(n_rows):
                row=[]
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images
  

#2-D --> 1-D flattening
def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

def extract_features(X):
    return [flatten_list(sample) for sample in X]


#calculating Euclidean distance
def dist(x,y):
    return sum(
        [(bytes_to_int(x_i)-bytes_to_int(y_i))**2 for x_i,y_i in zip(x,y)]
        )**(0.5)



def get_training_dist_for_test_sample(X_train,test_sample):
    return [dist(train_sample,test_sample) for train_sample in X_train]


def get_most_freq_ele(l):
    return max(l,key=l.count)

#knn
def knn(X_train,X_test,y_train,k=3):
    y_pred=[]
    for test_sample_idx,test_sample in enumerate(X_test):
        training_distances = get_training_dist_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [
            pair[0]
            for pair in 
            sorted(enumerate(training_distances),
            key=lambda x:x[1])
        ]
        #print(sorted_distance_indices)
        candidates = [
            bytes_to_int(y_train[idx])
            for idx in sorted_distance_indices[:k]
        ]
        top = get_most_freq_ele(candidates)
        y_pred.append(top)
    return y_pred
    #print(f'Point is {bytes_to_int(y_test[test_sample_idx])} and we guessed  {candidates}')


def main():
    X_train = read_images(TRAIN_DATA_FILENAME,15000) #60000
    #print(X_train[0]) --> 28 X 28 (2-D)
    
    y_train = read_labels(TRAIN_LABELS_FILENAME) #60000
    #print(y_train) --> (1-D)
    
    X_test = read_images(TEST_DATA_FILENAME,5) #10000
    y_test = read_labels(TEST_LABELS_FILENAME) #10000

    X_test=[read_image(f'{TEST_DIR}our_test.png')]
    
    for idx,test_sample in enumerate(X_test):
        write_image(test_sample,f'{TEST_DIR}{idx}.png')

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)
    #print(len(X_train[0]))-->784
    
    y_pred = knn(X_train,X_test,y_train)    
    print(y_pred)

if __name__ == '__main__':
    main()