%% How to use Melio Finder - run in terminal unless stated otherwise, GPU strongly recommended

1) provide coordinates of image centers in a .csv file (see data.csv for formatting)
2) run export.py in QGIS Python Console: exec(Path('C:/Users/JP/Melio_finder/export.py').read_text())
3) run RGB_to_grayscale.py to delete images from undefined layers and convert remaining images to grayscale
4) download trained model from: https://drive.google.com/file/d/1SvteK2lqISPKz8u2zzgx_WkN91Rdm2TH/view?usp=drive_link
4) run melio_finder.py to find images that contain meliorations


%% How to train Melio Finder (optional)

1) find training images
2) run create_csvs.py and then create_datasets.py to create 5 datasets for cross-validation
- assumes that number of images from a location is the same in both classes
3) run train_model.py and then fine_tune_model.py to train Melio Finder
4) run test_model.py to evaluate Melio Finder accuracy on labeled test data
