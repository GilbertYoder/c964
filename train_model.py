from fastai.vision.all import *

# The dataset is not included in this repository and 
# should be downloaded first:
# https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset
path = './dataset'

datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

dls = datablock.dataloaders(path)

learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fit_one_cycle(4)
learn.export('weather_classifier.pkl')