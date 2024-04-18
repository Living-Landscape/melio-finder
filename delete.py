# delete images from undefined layers

import cv2
import os
import numpy as np
import csv

# load data
file = open("data.csv")
reader = csv.reader(file)
data = list(reader)

# delete images
for index in range(1, len(data)):
    for year in range(1998, 2020):
        file = "img_%s_%s.png" % (index, year)
        if os.path.isfile(file):
            os.remove("img_%s_%s.png.aux.xml" % (index, year))  # delete aux.xml file
            img = cv2.imread("img_%s_%s.png" % (index, year))

            if np.size(img) * 250 < np.sum(img):  # if true => layer is not defined (white image with watermarks)
                # delete image
                os.remove("img_%s_%s.png" % (index, year))
