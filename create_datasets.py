# create 5 datasets for cross-validation
import shutil
import random
import csv
import math
import os

List = []
Y_file = open("Y.csv")
Y_reader = csv.reader(Y_file)
Y_data = list(Y_reader)
N_file = open("N.csv")
N_reader = csv.reader(N_file)
N_data = list(N_reader)

# create directories
for d in range(1, 6):
    try:
        os.makedirs("%s/N")
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs("%s/Y")
    except FileExistsError:
        # directory already exists
        pass

# copy images to datasets
for i in range(1, len(N_data)+1):
    List.append(i)
random.shuffle(List)

for i in range(1, len(N_data)+1):  # assumes same number of N and Y images
    num = List[i-1]
    Y_index = Y_data[num-1][1]
    Y_year = Y_data[num-1][2]
    N_index = N_data[num-1][1]
    N_year = N_data[num-1][2]
    k = math.floor((i-1)/math.floor(len(N_data)/5)) + 1
    if k == 6:  # put remaining 1-4 images into 5th dataset
        k = 5
    for kk in range(1, 6):
        if ((kk - k) % 5) == 0:
            shutil.copy('training_images/Y/img_%s_%s.png' % (Y_index, Y_year),
                        '%s/validation/Y/img_%s_%s.png' % (kk, Y_index, Y_year))
            shutil.copy('training_images_grayscale/Y/img_%s_%s.png' % (Y_index, Y_year),
                        '%s_g/validation/Y/img_%s_%s.png' % (kk, Y_index, Y_year))
            shutil.copy('training_images/N/img_%s_%s.png' % (N_index, N_year),
                        '%s/validation/N/img_%s_%s.png' % (kk, N_index, N_year))
            shutil.copy('training_images_grayscale/N/img_%s_%s.png' % (N_index, N_year),
                        '%s_g/validation/N/img_%s_%s.png' % (kk, N_index, N_year))
        else:
            shutil.copy('training_images/Y/img_%s_%s.png' % (Y_index, Y_year),
                        '%s/training/Y/img_%s_%s.png' % (kk, Y_index, Y_year))
            shutil.copy('training_images_grayscale/Y/img_%s_%s.png' % (Y_index, Y_year),
                        '%s_g/training/Y/img_%s_%s.png' % (kk, Y_index, Y_year))
            shutil.copy('training_images/N/img_%s_%s.png' % (N_index, N_year),
                        '%s/training/N/img_%s_%s.png' % (kk, N_index, N_year))
            shutil.copy('training_images_grayscale/N/img_%s_%s.png' % (N_index, N_year),
                        '%s_g/training/N/img_%s_%s.png' % (kk, N_index, N_year))
