import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import math

import common
import config as cfg
import model

import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

# Mettre cette variable à True si vous voulez voir le résultat sur la base de test
on_test=False

images, labels, labels2, list_labels=common.read_xmls(cfg.dir_dataset, 1)
nbr_classes=len(list_labels)

if on_test:
  images, labels, labels2_test, list_labels_test=common.read_xmls(cfg.dir_test,
                                                                  1,
                                                                  list_labels=list_labels,
                                                                  verbose=True)

images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)
index=np.random.permutation(len(images))
images=images[index]
labels=labels[index]

model=model.model(nbr_classes, cfg.nbr_boxes, cfg.cellule_y, cfg.cellule_x)

checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(cfg.dir_model))

def inference(image, model):
  grid=np.meshgrid(np.arange(cfg.cellule_x, dtype=np.float32), np.arange(cfg.cellule_y, dtype=np.float32))
  grid=np.expand_dims(np.stack(grid, axis=-1), axis=2)
  grid=np.tile(grid, (1, 1, cfg.nbr_boxes, 1))

  predictions=model(np.array([image]))
  pred_boxes=predictions[0, :, :, :, 0:4]
  pred_conf=common.sigmoid(predictions[0, :, :, :, 4])
  pred_classes=common.softmax(predictions[0, :, :, :, 5:5+nbr_classes])
  ids=np.argmax(pred_classes, axis=-1)

  x_center=((grid[:, :, :, 0]+common.sigmoid(pred_boxes[:, :, :, 0]))*cfg.r_x)
  y_center=((grid[:, :, :, 1]+common.sigmoid(pred_boxes[:, :, :, 1]))*cfg.r_y)
  w=(np.exp(pred_boxes[:, :, :, 2])*cfg.anchors[:, 0]*cfg.r_x)
  h=(np.exp(pred_boxes[:, :, :, 3])*cfg.anchors[:, 1]*cfg.r_y)

  x_min=(x_center-w/2).astype(np.int32)
  y_min=(y_center-h/2).astype(np.int32)
  x_max=(x_center+w/2).astype(np.int32)
  y_max=(y_center+h/2).astype(np.int32)
    
  tab_boxes=np.stack([y_min, x_min, y_max, x_max], axis=-1).reshape(-1, 4).astype(np.float32)
  pred_conf=pred_conf.reshape(-1)
  ids=ids.reshape(-1)
  tab_index=tf.image.non_max_suppression(tab_boxes, pred_conf, 42)

  return tab_boxes, pred_conf, ids, tab_index

for i in range(len(images)):
  img2=images[i].copy()
  img=common.prepare_image(images[i], labels[i], False)
  
  tab_boxes, pred_conf, ids, tab_index=inference(images[i], model)
    
  for id in tab_index:
    if pred_conf[id]>cfg.seuil_conf:

      x_min=tab_boxes[id, 1]
      y_min=tab_boxes[id, 0]
      x_max=tab_boxes[id, 3]
      y_max=tab_boxes[id, 2]

      color=cfg.colors[ids[id]]
      
      cv2.rectangle(img2, (x_min, y_min), (x_max, y_max), color, 1)
      cv2.rectangle(img2, (x_min, y_min), (x_max, int(y_min-15)), color, cv2.FILLED)
      cv2.putText(img2, "{:3.0%}".format(pred_conf[id]), (x_min, int(y_min-5)), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (255, 255, 255), 1) # {%} ???

  cv2.imshow("Bonne reponse (dataset)", img)
  cv2.imshow("Inference", img2)
    
  key=cv2.waitKey()&0xFF
  if key==ord('q'):
    quit()
