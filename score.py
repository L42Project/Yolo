import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import math

import common
import config as cfg
import model

def calcul_map(model, dataset, nbr_classes, labels2, seuil_iou=0.5, beta=1., confidence_min=0.1, verbose=False):
  grid=np.meshgrid(np.arange(cfg.cellule_x, dtype=np.float32), np.arange(cfg.cellule_y, dtype=np.float32))
  grid=np.expand_dims(np.stack(grid, axis=-1), axis=2)
  grid=np.tile(grid, (1, 1, 1, cfg.nbr_boxes, 1))

  index_labels2=0
  labels2_=labels2*[cfg.r_x, cfg.r_y, cfg.r_x, cfg.r_y, 1, 1, 1]
  score=[]
  tab_nbr_reponse=[]
  tab_tp=[]
  tab_true_boxes=[]

  for images, labels in dataset:
    predictions=np.array(model(images))

    pred_conf=common.sigmoid(predictions[:, :, :, :, 4])
    pred_classes=common.softmax(predictions[:, :, :, :, 5:5+nbr_classes])
    pred_ids=np.argmax(pred_classes, axis=-1)

    x_center=((grid[:, :, :, :, 0]+common.sigmoid(predictions[:, :, :, :, 0]))*cfg.r_x)
    y_center=((grid[:, :, :, :, 1]+common.sigmoid(predictions[:, :, :, :, 1]))*cfg.r_y)
    w=(np.exp(predictions[:, :, :, :, 2])*cfg.anchors[:, 0]*cfg.r_x)
    h=(np.exp(predictions[:, :, :, :, 3])*cfg.anchors[:, 1]*cfg.r_y)

    x_min=x_center-w/2
    y_min=y_center-h/2
    x_max=x_center+w/2
    y_max=y_center+h/2

    tab_boxes=np.stack([y_min, x_min, y_max, x_max], axis=-1).astype(np.float32)
    tab_boxes=tab_boxes.reshape(-1, cfg.cellule_y*cfg.cellule_x*cfg.nbr_boxes, 4)
    pred_conf=pred_conf.reshape(-1, cfg.cellule_y*cfg.cellule_x*cfg.nbr_boxes)
    pred_ids=pred_ids.reshape(-1, cfg.cellule_y*cfg.cellule_x*cfg.nbr_boxes)

    for p in range(len(predictions)):
      nbr_reponse=np.zeros(nbr_classes)
      tp=np.zeros(nbr_classes)
      nbr_true_boxes=np.zeros(nbr_classes)
      tab_index=tf.image.non_max_suppression(tab_boxes[p], pred_conf[p], 100)      
      for id in tab_index:
        if pred_conf[p, id]>confidence_min:
          nbr_reponse[pred_ids[p, id]]+=1
          for box in labels2_[index_labels2]:
            if not box[5]:
              break
            b1=[tab_boxes[p, id, 1], tab_boxes[p, id, 0], tab_boxes[p, id, 3], tab_boxes[p, id, 2]]
            iou=common.intersection_over_union(b1, box)
            if iou>seuil_iou and box[6]==pred_ids[p, id]:
              tp[pred_ids[p, id]]+=1

      for box in labels2[index_labels2]:
        if not box[5]:
          break
        nbr_true_boxes[int(box[6])]+=1

      tab_nbr_reponse.append(nbr_reponse)
      tab_tp.append(tp)
      tab_true_boxes.append(nbr_true_boxes)
      
      index_labels2=index_labels2+1

  tab_nbr_reponse=np.array(tab_nbr_reponse)
  tab_tp=np.array(tab_tp)
  tab_true_boxes=np.array(tab_true_boxes)

  precision=tab_tp/(tab_nbr_reponse+1E-7)
  rappel=tab_tp/(tab_true_boxes+1E-7)

  score=np.mean((1+beta*beta)*precision*rappel/(beta*beta*precision+rappel+1E-7))

  if verbose:
    print("Précision: {:.2%}".format(np.mean(precision)))
    print("Rappel   : {:.2%}".format(np.mean(rappel)))
    print("Score    : {:.2%}".format(score))

  return score

