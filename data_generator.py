# -*- coding: utf-8 -*-
import os
import random
from random import shuffle

import cv2 as cv
import numpy as np
import sklearn.neighbors as nn
from keras.utils import Sequence

from config import batch_size, img_rows, img_cols, nb_neighbors

#image_folder = '/mnt/code/ImageNet-Downloader/image/resized'
pacs_dir = './pacs_data'

def get_soft_encoding(image_ab, nn_finder, nb_q):
    h, w = image_ab.shape[:2]
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    ab = np.vstack((a, b)).T
    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
    # Smooth the weights with a gaussian kernel
    sigma_neighbor = 5
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    # format the tar get
    y = np.zeros((ab.shape[0], nb_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, idx_neigh] = wts
    y = y.reshape(h, w, nb_q)
    return y


class DataGenSequence(Sequence):
    def __init__(self, usage, domain=None, name_dir="./", image_dir="./voc"):

        # domain은 list
        self.usage = usage
        self.image_dir = image_dir

        if usage == 'train':
            if domain is not None:
                names_files = ['%s_train_names.txt'%i for i in domain]
            else:
                names_file = 'train_names.txt'
        else:
            if domain is not None:
                names_files = ['%s_valid_names.txt'%i for i in domain]
            else:
                names_file = 'valid_names.txt'

        if domain is not None:
            self.names = []
            for name_file in names_files:
                with open(os.path.join(name_dir,name_file), 'r') as f:
                    self.names += f.read().splitlines()
        else:
            with open(os.path.join(name_dir, names_file), 'r') as f:
                self.names = f.read().splitlines()

        np.random.shuffle(self.names)
        print("number of name file", len(self.names), self.names[0])
        # Load the array of quantized ab value
        q_ab = np.load("data/pts_in_hull.npy")
        self.nb_q = q_ab.shape[0]
        # Fit a NN to q_ab
        self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        out_img_rows, out_img_cols = img_rows // 4, img_cols // 4

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 1), dtype=np.float32)
        batch_y = np.empty((length, out_img_rows, out_img_cols, self.nb_q), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            filename = os.path.join(self.image_dir, name)
            # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
            bgr = cv.imread(filename)
            # bgr = cv.resize(bgr, (img_rows, img_cols), cv.INTER_CUBIC)
            # like infrared 
            gray = bgr[:, :, :2] * 0
            gray = cv.cvtColor(gray, cv.COLOR_BGR2Lab)
            gray = gray[:,:, 0]
            # gray = cv.resize(gray, (img_rows, img_cols), cv.INTER_CUBIC)
            lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
            x = gray / 255.
            x = cv.resize(x, (img_rows, img_cols), cv.INTER_CUBIC)
            
            out_lab = cv.resize(lab, (out_img_rows, out_img_cols), cv.INTER_CUBIC)
            # Before: 42 <=a<= 226, 20 <=b<= 223
            # After: -86 <=a<= 98, -108 <=b<= 95
            out_ab = out_lab[:, :, 1:].astype(np.int32) - 128

            y = get_soft_encoding(out_ab, self.nn_finder, self.nb_q)

            if np.random.random_sample() > 0.5:
                x = np.fliplr(x)
                y = np.fliplr(y)

            batch_x[i_batch, :, :, 0] = x
            batch_y[i_batch] = y

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen(domain=None, name_dir="", image_dir=""):
    return DataGenSequence('train', domain, name_dir, image_dir)


def valid_gen(domain=None, name_dir="", image_dir=""):
    return DataGenSequence('valid', domain, name_dir, image_dir)


# +
def split_data(image_folder, domain='photo', include_test=False, store_dir='./'):
    os.makedirs(store_dir, exist_ok=True)
    names = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    num_samples = len(names)  
    print('num_samples: ' + str(num_samples))

    if include_test:
        num_train_samples = int(num_samples * 0.982)
        num_test_samples = int(num_samples * 0.01)
    else:
        num_train_samples = int(num_samples * 0.98)
        num_test_samples = 0
    print('num_train_samples: ' + str(num_train_samples))
    print('num_test_samples: ' + str(num_test_samples))
    num_valid_samples = num_samples - num_train_samples - num_test_samples
    print('num_valid_samples: ' + str(num_valid_samples))
    valid_names = random.sample(names, num_valid_samples)
    test_names = []
    names_wo_valid = [n for n in names if n not in valid_names]
    if include_test:
        test_names = random.sample(names_wo_valid, num_test_samples)
        shuffle(test_names)
        with open(os.path.join(store_dir, 'test_names.txt'), 'w') as file:
            file.write('\n'.join(test_names))
    train_names = [n for n in names_wo_valid if n not in test_names]
    shuffle(valid_names)
    shuffle(train_names)

    # with open('names.txt', 'w') as file:
    #     file.write('\n'.join(names))

    with open(os.path.join(store_dir, 'valid_names.txt'), 'w') as file:
        file.write('\n'.join(valid_names))

    with open(os.path.join(store_dir, 'train_names.txt'), 'w') as file:
        file.write('\n'.join(train_names))
        
        
def get_names(name_dir, mode='train'):
    names = []
    with open(os.path.join(name_dir, '%s.txt'%mode), 'r') as f:
        names += f.read().splitlines()
    
    for i in range(len(names)):
        names[i] = names[i] + '.jpg'
        
    with open(os.path.join(name_dir, '%s_names.txt'%mode), 'w') as file:
        file.write('\n'.join(names))
        


# -

if __name__ == '__main__':
    split_data('./VOC2012/JPEGImages', None, False, './VOC2012')

