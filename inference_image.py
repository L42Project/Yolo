import cv2
import tensorflow as tf
import sys
import time
import os
import glob
import numpy as np

import common
import config as cfg

dir_image="dataset/dataset1/"

model=common.readmodel()
if model is None:
    quit()

list_labels=['tete', 'humain']
nbr_classes=len(list_labels)

list_attributs=None
nbr_attributs=0

for image in glob.glob(dir_image+"/*.jpg"):
    image=cv2.imread(image)

    tab_boxes, pred_conf, ids, pred_attr, tab_index=common.inference(image, model, nbr_classes, nbr_attributs)

    for id in tab_index:
        if pred_conf[id]>cfg.seuil_conf:

            y_min, x_min, y_max, x_max=tab_boxes[id]

            color=cfg.colors[ids[id]]
            color=np.array(color)*255

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)
            cv2.rectangle(image, (x_min, y_min), (x_max, int(y_min-15)), color, cv2.FILLED)
            cv2.putText(image, "{:3.0%}".format(pred_conf[id]), (x_min, int(y_min-5)), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (255, 255, 255), 1)

    cv2.imshow("Inference", image)

    key=cv2.waitKey()&0xFF
    if key==ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        quit()
