from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg #to laod the config of ig model
from detectron2 import model_zoo #to laod pre-trained weights form model zoo
from detectron2.utils.visualizer import ColorMode
import random
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import ImageOps
import numpy as np
from scipy.spatial import distance as dist

canny_path = "canny_out.jpg"
convexhull_path = "ConvexHull.jpg"
opening_path = "opening.jpg"

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    if len(cnts) == 2:
        cnts = cnts[0]
    # if the length of the contours tuple is '3' then we are using
    elif len(cnts) == 3:
        cnts = cnts[1]
    # otherwise OpenCV has changed their cv2.findContours return
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))
    # return the actual contours array
    return cnts
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def plot_samples(dataset_name, n=1): #check if the annotations are correct
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
    cfg.SOLVER.MAX_ITER = 5000
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
    v = Visualizer(im[:,:,::-1], metadata={}, scale= 0.4, instance_mode = ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("predictor_output.jpg", v.get_image())
    
    
def convex_hull(canny_path):
    img1 = cv2.imread('canny_out.jpg')
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img,50,255,0)
    # Find the contours
    h = img1.shape[0]
    w = img1.shape[1]
    img_mask = np.zeros([h, w, 3], np.uint8)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # For each contour, find the convex hull and draw it
    # on the original image.
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        cv2.drawContours(img_mask, [hull], -1, (255, 0, 0), 5)
    # Display the final convex hull image
    cv2.imwrite('ConvexHull.jpg',img_mask)
    
def fill_boundary(convexhull_path):
    image = cv2.imread('ConvexHull.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(gray,[c], 0, (255,255,255), -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)

    cv2.imwrite('opening.jpg', opening)
    
def transformations(opening_path, image_path):
    orig = cv2.imread(opening_path)
    orig1 = cv2.imread(image_path)
    image = orig.copy()
    image1 = orig1.copy()
    image = resize(image, width=500)
    image1 = resize(image1, width=500)
    cv2.imwrite('ConvexHull_mask.jpg', image)
    cv2.imwrite('original_image.jpg', image1)
    ratio = orig.shape[1] / float(image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    receiptCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the receipt
        if len(approx) == 4:
            receiptCnt = approx
            break
        # if the receipt contour is empty then our script could not find the
        # outline and we should be notified
    print(receiptCnt)
    if receiptCnt is None:
        raise Exception(("Could not find receipt outline. "
            "Try debugging your edge detection and contour steps."))
            
    receipt = four_point_transform(orig1, receiptCnt.reshape(4, 2) * ratio)
    cv2.imwrite('tranformed_output.jpg', resize(receipt, width=500))
    
    
    
def on_image1(image_path, predictor):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1], metadata={}, scale= 0.5, instance_mode = ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("predictor_output6.jpg", v.get_image())
    