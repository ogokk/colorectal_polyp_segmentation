

import cv2
import glob

mask_path = "C:/Users/yourpath/datasets/masks/"

masks = [cv2.imread(file,0) for file in glob.glob('C:/Users/yourpath/datasets/masks/*.tiff')]

zeros = []
ones = []

for i in range(len(masks)):
    ones.append((masks[i] == 255).sum())
    zeros.append((masks[i] == 0).sum())
    
total_ones = sum(ones)
total_zeros = sum(zeros)

total = total_ones + total_zeros

percentage_ones  = total_ones/total
percentage_zeros = total_zeros/total 
    
