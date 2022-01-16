from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
import random
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import ImageOps
import numpy as np

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    
    for s in random.sample(dataset_custom, n):
        print('s : ', s)
        image_path = s['file_name']
        head_tail = os.path.split(image_path)
        tail = head_tail[1]
        print('image_path :', image_path)
        img = cv2.imread(image_path)
        print('img : ', img)
        v = Visualizer(img[:,:,::-1], metadata=dataset_custom_metadata, scale = 0.4)
        v = v.draw_dataset_dict(s)
        cv2.imwrite(tail, v.get_image())
        
def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    
    cfg.DATALOADER.NUM_WORKERS = 2
    
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    
    cfg.OUTPUT_DIR = output_dir
    
    return cfg
    
def on_image(image_path, predictor):
    im = cv2.imread(image_path)
    head_tail = os.path.split(image_path)
    tail = head_tail[1]
    outputs = predictor(im)
    mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
    num_instances = mask_array.shape[0]
    num_instance = mask_array.shape
    print('num_instance : ',num_instance)
    scores = outputs['instances'].scores.to("cpu").numpy()
    labels = outputs['instances'].pred_classes .to("cpu").numpy()
    bbox   = outputs['instances'].pred_boxes.to("cpu").tensor.numpy()

    mask_array = np.moveaxis(mask_array, 0, -1)

    mask_array_instance = []
    #img = np.zeros_like(im) #black
    h = im.shape[0]
    w = im.shape[1]
    img_mask = np.zeros([h, w, 3], np.uint8)
    color = (200, 100, 255)
    for i in range(num_instances):
        #img = np.zeros_like(im)
        temp = cv2.imread(image_path)
        for j in range(temp.shape[2]):
            temp[:,:,j] = temp[:,:,j] * mask_array[:,:,i]
        cv2.imwrite("mask_roi.jpg", temp[:,:,j])   
    for i in range(num_instances):
        img = np.zeros_like(im)
        mask_array_instance.append(mask_array[:, :, i:(i+1)])
        img = np.where(mask_array_instance[i] == True, 255, img)
        array_img = np.asarray(img)
        img_mask[np.where((array_img==[255,255,255]).all(axis=2))]=color
    img_mask = np.asarray(img_mask)
    cv2.imwrite("mask_out.jpg", img_mask)
    edges = cv2.Canny(image=img_mask, threshold1=100, threshold2=200)
    cv2.imwrite("canny_out.jpg", edges)
    img1 = cv2.imread('canny_out.jpg')
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    ret, thresh = cv2.threshold(img,50,255,0)
    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # For each contour, find the convex hull and draw it
    # on the original image.
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        cv2.drawContours(img1, [hull], -1, (255, 0, 0), 2)
    # Display the final convex hull image
    cv2.imwrite('ConvexHull.jpg', img1)
    v = Visualizer(im[:,:,::-1], metadata={}, scale= 0.4, instance_mode = ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("test_out.jpg", v.get_image())     