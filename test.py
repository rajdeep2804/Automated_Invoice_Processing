from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

cfg_save_path = "IS_cfg.pickle"


with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)
    
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
predictor = DefaultPredictor(cfg)

image_path = "data/test/IMG_3708 5.18.41 AM.jpg"
canny_path = "canny_out.jpg"
convexhull_path = "ConvexHull.jpg"
opening_path = "opening.jpg"

on_image(image_path, predictor)
convex_hull(canny_path)
fill_boundary(convexhull_path)
transformations(opening_path, image_path)
