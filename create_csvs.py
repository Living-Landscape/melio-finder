import csv
import os
outputFile = open('Y.csv', 'w', newline='')
outputWriter = csv.writer(outputFile)
counter = 0
for index in range(1, 6138):
    for year in range(1998, 2020):
        file = "training_images/Y/img_%s_%s.png" % (index, year)
        # check if file exists !!!
        if os.path.isfile(file):
            counter = counter + 1
            outputWriter.writerow([counter, index, year])
outputFile.close()

outputFile = open('N.csv', 'w', newline='')
outputWriter = csv.writer(outputFile)
counter = 0
for index in range(1, 6138):
    for year in range(1998, 2020):
        file = "training_images/N/img_%s_%s.png" % (index, year)
        # check if file exists !!!
        if os.path.isfile(file):
            counter = counter + 1
            outputWriter.writerow([counter, index, year])
outputFile.close()
