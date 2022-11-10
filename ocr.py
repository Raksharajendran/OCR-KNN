# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:41:33 2022

@author: Raksha Rajendran
"""

#OCR using KNN

#filenames
DATA_DIR = 'data/'
TEST_DATA_FILENAME=DATA_DIR+'t10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR+'t10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME =DATA_DIR + 'train-labels.idx1-ubyte'


#conversion
def bytes_to_int(byte_data):
    return int.from_bytes(byte_data,'big')


#reading labels of traing data
def read_labels(filename,n_max_labels=None):
    labels=[]
    with open(filename,'rb') as f:
        _ = bytes_to_int(f.read(4))
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels

#reading images of training data 
def read_images(filename,n_max_images=None):
    images=[]
    with open(filename,'rb') as f:
        _ = bytes_to_int(f.read(4))
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
                    pixel = bytes_to_int(f.read(1))
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
        [(x_i-y_i)**2 for x_i,y_i in zip(x,y)]
        )**(0.5)



def get_training_dist_for_test_sample(X_train,test_sample):
    return [dist(train_sample,test_sample) for train_sample in X_train]


#knn
def knn(X_train,X_test,y_train,y_test,k=3):
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
            y_train[idx]
            for idx in sorted_distance_indices[:k]
        ]
        print(f'Point is {y_test[test_sample_idx]} and we guessed {candidates}')


def main():
    X_train = read_images(TRAIN_DATA_FILENAME,15000) #60000
    #print(X_train[0]) --> 28 X 28 (2-D)
    
    y_train = read_labels(TRAIN_LABELS_FILENAME) #60000
    #print(y_train) --> (1-D)
    
    X_test = read_images(TEST_DATA_FILENAME,5) #10000
    y_test = read_labels(TEST_LABELS_FILENAME) #10000

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)
    #print(len(X_train[0]))-->784
    
    y_pred = knn(X_train,X_test,y_train,y_test)    


if __name__ == '__main__':
    main()