# export images - run in QGIS Python Console

import subprocess
import csv

# read coordinates of image centers from csv file
file = open("C:/Users/JP/Melio_finder/data.csv")
reader = csv.reader(file)
data = list(reader)

# parameters - resolution and image size - do not change
tr = 0.3
win = 150

# export images
for index in range(1, len(data)):

    x = data[index][0]
    y = data[index][1]

    # export from current orthophotomap

    subprocess.call('gdal_translate -tr %s %s -projwin %s %s %s %s -projwin_srs EPSG:5514 -of PNG '
                    '"WMS:http://geoportal.cuzk.cz/WMS_ORTOFOTO_PUB/service.svc/get?SERVICE=WMS&VERSION=1.1.1&REQUEST'
                    '=GetMap&LAYERS=GR_ORTFOTORGB&SRS=EPSG:900913&BBOX=1327926.342591154389083385,'
                    '6145626.423431359231472015,2112791.728672875091433525,6684240.236736821010708809" '
                    '"C:/Users/JP/Melio_finder/img_%s_2019.png"'
                    % (tr, tr, float(x)-win, float(y)+win, float(x)+win, float(y)-win, index))

    # export from archive orthophotomaps

    for year in range(1998, 2019):
        subprocess.call('gdal_translate -tr %s %s -projwin %s %s %s %s -projwin_srs EPSG:5514 -of PNG '
                        '"WMS:http://geoportal.cuzk.cz/WMS_ORTOFOTO_ARCHIV/service.svc/get?SERVICE=WMS&VERSION=1.1.1'
                        '&REQUEST=GetMap&LAYERS=%s&SRS=EPSG:900913&BBOX=1327926.342591154389083385,'
                        '6145626.423431359231472015,2112791.728672875091433525,6684240.236736821010708809" "C:/Users/'
                        'JP/Melio_finder/img_%s_%s.png"'
                        % (tr, tr, float(x)-win, float(y)+win, float(x)+win, float(y)-win, year, index, year))
