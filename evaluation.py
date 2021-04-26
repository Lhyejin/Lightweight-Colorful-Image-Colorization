import numpy
import math
import cv2
import pandas as pd
import os
import argparse
from skimage.metrics import structural_similarity as ssim
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def psnr (original, colorized) :
    mse = numpy.mean((original-colorized)**2)
    if mse == 0 :
        return 100
    else :
        return 20*math.log10(255.0/math.sqrt(mse))



parser = argparse.ArgumentParser()
parser.add_argument("--original", default='./images/origin', type=str)
parser.add_argument("--color", default='./images/colorized', type=str, help="Test dir path")
parser.add_argument("--output", default='result', type=str)
args = parser.parse_args()
print(args)
original_path = args.original
colorized_path = args.color

original_list = [os.path.join(original_path, f) for f in os.listdir(original_path) if os.path.isfile(os.path.join(original_path, f))]
colorized_list = [os.path.join(colorized_path, f) for f in os.listdir(colorized_path) if os.path.isfile(os.path.join(colorized_path, f))]
original_list.sort()
colorized_list.sort()
print(original_list)
print(colorized_list)

psnr_list = []
ssim_list = []



for i in range(len(colorized_list)):
    origin_img = cv2.imread(original_list[i])
    colorized_img = cv2.imread(colorized_list[i])
    origin_img = cv2.resize(origin_img, dsize=(colorized_img.shape[0], colorized_img.shape[1]))
    psnr_list.append(psnr(origin_img, colorized_img))
    ssim_list.append(ssim(origin_img, colorized_img, data_range=colorized_img.max() - colorized_img.min(), multichannel=True))
    print(psnr_list[i])
    print(ssim_list[i])

print("mean", numpy.mean(psnr_list))
print("ssim mean", numpy.mean(ssim_list))
print("")
data = {'original' : original_list,
        'colorized' : colorized_list,
        'psnr': psnr_list,
        'ssim': ssim_list,
        'psnr_mean': numpy.mean(psnr_list),
        'ssim_mean': numpy.mean(ssim_list)
        }

df = pd.DataFrame(data)
df.to_csv("%s.csv"%(args.output), mode='w')
