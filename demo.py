# import the necessary packages
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2 as cv
import keras.backend as K
import numpy as np
import sklearn.neighbors as nn

from config import img_rows, img_cols
from config import nb_neighbors, T, epsilon
import model
import separable_model
import half_separable_model
import argparse

# +
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='', type=str)
    parser.add_argument("--model-type", default='original', type=str, help="original, separable, half")
    parser.add_argument("--testset-path", default='./VOC2012/testset', type=str)
    parser.add_argument("--name-path", default='./VOC2012/test_names.txt', type=str)
    args = parser.parse_args()
    print(args)
    channel = 3
    
    model_weights_path = args.model_path
    model = model.build_model()
    if args.model_type == "separable":
        model = separable_model.build_model()
    elif args.model_type == "half":
        model = half_separable_mode.build_model()

    model.load_weights(model_weights_path)

#     print(model.summary())

    image_folder = args.testset_path
    names_file = args.name_path
    with open(names_file, 'r') as f:
        names = f.read().splitlines()
        
    samples = random.sample(names, 500)

    h, w = img_rows // 4, img_cols // 4

    # Load the array of quantized ab value
    q_ab = np.load("data/pts_in_hull.npy")
    nb_q = q_ab.shape[0]

    # Fit a NN to q_ab
    nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    for i in range(len(samples)):
        image_name = names[i]
        filename = os.path.join(image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
        bgr = cv.imread(filename)
        # like infrared
        gray = bgr[:, :, :2] * 0
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2Lab)
        gray = gray[:,:, 0]
        bgr = cv.resize(bgr, (img_rows, img_cols), cv.INTER_CUBIC)
        gray = cv.resize(gray, (img_rows, img_cols), cv.INTER_CUBIC)
        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]
        # print('np.max(L): ' + str(np.max(L)))
        # print('np.min(L): ' + str(np.min(L)))
        # print('np.max(a): ' + str(np.max(a)))
        # print('np.min(a): ' + str(np.min(a)))
        # print('np.max(b): ' + str(np.max(b)))
        # print('np.min(b): ' + str(np.min(b)))
        x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = gray / 255.

        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        X_colorized = model.predict(x_test)
        X_colorized = X_colorized.reshape((h * w, nb_q))

        # Reweight probas
        X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
        X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

        # Reweighted
        q_a = q_ab[:, 0].reshape((1, 313))
        q_b = q_ab[:, 1].reshape((1, 313))

        X_a = np.sum(X_colorized * q_a, 1).reshape((h, w))
        X_b = np.sum(X_colorized * q_b, 1).reshape((h, w))
        # print('np.max(X_a): ' + str(np.max(X_a)))
        # print('np.min(X_a): ' + str(np.min(X_a)))
        # print('np.max(X_b): ' + str(np.max(X_b)))
        # print('np.min(X_b): ' + str(np.min(X_b)))
        X_a = cv.resize(X_a, (img_rows, img_cols), cv.INTER_CUBIC)
        X_b = cv.resize(X_b, (img_rows, img_cols), cv.INTER_CUBIC)

        # Before: -90 <=a<= 100, -110 <=b<= 110
        # After: 38 <=a<= 228, 18 <=b<= 238
        X_a = X_a + 128
        X_b = X_b + 128
        # print('np.max(X_a): ' + str(np.max(X_a)))
        # print('np.min(X_a): ' + str(np.min(X_a)))
        # print('np.max(X_b): ' + str(np.max(X_b)))
        # print('np.min(X_b): ' + str(np.min(X_b)))

        out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
        out_lab[:, :, 0] = lab[:, :, 0]
        out_lab[:, :, 1] = X_a
        out_lab[:, :, 2] = X_b
        out_L = out_lab[:, :, 0]
        out_a = out_lab[:, :, 1]
        out_b = out_lab[:, :, 2]
        # print('np.max(out_L): ' + str(np.max(out_L)))
        # print('np.min(out_L): ' + str(np.min(out_L)))
        # print('np.max(out_a): ' + str(np.max(out_a)))
        # print('np.min(out_a): ' + str(np.min(out_a)))
        # print('np.max(out_b): ' + str(np.max(out_b)))
        # print('np.min(out_b): ' + str(np.min(out_b)))
        out_lab = out_lab.astype(np.uint8)
        out_bgr = cv.cvtColor(out_lab, cv.COLOR_LAB2BGR)
        # print('np.max(out_bgr): ' + str(np.max(out_bgr)))
        # print('np.min(out_bgr): ' + str(np.min(out_bgr)))
        out_bgr = out_bgr.astype(np.uint8)

        
        os.makedirs('images/gray/%s'%(args.model_type), exist_ok=True)
        os.makedirs('images/gt/%s'%(args.model_type), exist_ok=True)
        os.makedirs('images/colorized/%s'%(args.model_type), exist_ok=True)
        cv.imwrite('images/gray/{}/{}.png'.format(args.model_type, i), gray)
        cv.imwrite('images/gt/{}/{}.png'.format(args.model_type, i), bgr)
        cv.imwrite('images/colorized/{}/{}.png'.format(args.model_type, i), out_bgr)

    K.clear_session()
