from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

cfg_save_path = "IS_cfg.pickle"


with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)
    
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #load model weight path of custom dataset we trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
predictor = DefaultPredictor(cfg)

image_path = "image_path"
on_image(image_path, predictor)
convex_hull(canny_path)
fill_boundary(convexhull_path)
transformations(opening_path, image_path)
#on_image1(image_path, predictor)