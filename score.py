import tensorflow as tf
import sys
import time
import cv2
import numpy as np

import common
import config as cfg

def calcul_map(model, dataset, list_labels, list_attributs=None, seuil_iou=0.5, beta=1., confidence_min=0.1, verbose=False):
  grid=np.meshgrid(np.arange(cfg.nbr_cellule, dtype=np.float32), np.arange(cfg.nbr_cellule, dtype=np.float32))
  grid=np.expand_dims(np.stack(grid, axis=-1), axis=2)
  grid=np.tile(grid, (1, 1, 1, cfg.nbr_boxes, 1))

  nbr_classes=len(list_labels)
  
  score=[]
  tab_nbr_reponse=[]
  tab_tp=[]
  tab_true_boxes=[]
  score_attr=[]

  for images, labels, labels2, masks_attr in dataset:
    labels2_=labels2*[cfg.cellule_size, cfg.cellule_size, cfg.cellule_size, cfg.cellule_size, 1, 1, 1]
    predictions=np.array(model(images))
    pred_conf=common.sigmoid(predictions[:, :, :, :, 4])
    pred_classes=common.softmax(predictions[:, :, :, :, 5:5+nbr_classes])
    pred_ids=np.argmax(pred_classes, axis=-1)

    x_center=((grid[:, :, :, :, 0]+common.sigmoid(predictions[:, :, :, :, 0]))*cfg.cellule_size)
    y_center=((grid[:, :, :, :, 1]+common.sigmoid(predictions[:, :, :, :, 1]))*cfg.cellule_size)
    w=(np.exp(predictions[:, :, :, :, 2])*cfg.anchors[:, 0]*cfg.cellule_size)
    h=(np.exp(predictions[:, :, :, :, 3])*cfg.anchors[:, 1]*cfg.cellule_size)

    x_min=x_center-w/2
    y_min=y_center-h/2
    x_max=x_center+w/2
    y_max=y_center+h/2

    tab_boxes=np.stack([y_min, x_min, y_max, x_max], axis=-1).astype(np.float32)
    tab_boxes=tab_boxes.reshape(-1, cfg.nbr_cellule*cfg.nbr_cellule*cfg.nbr_boxes, 4)
    pred_conf=pred_conf.reshape(-1, cfg.nbr_cellule*cfg.nbr_cellule*cfg.nbr_boxes)
    pred_ids=pred_ids.reshape(-1, cfg.nbr_cellule*cfg.nbr_cellule*cfg.nbr_boxes)

    for p in range(len(predictions)):
      nbr_reponse=np.zeros(nbr_classes)
      tp=np.zeros(nbr_classes)
      nbr_true_boxes=np.zeros(nbr_classes)
      tab_index=tf.image.non_max_suppression(tab_boxes[p], pred_conf[p], 100)      
      for id in tab_index:
        if pred_conf[p, id]>confidence_min:
          nbr_reponse[pred_ids[p, id]]+=1
          for box in labels2_[p]:
            if not box[5]:
              break
            b1=[tab_boxes[p, id, 0], tab_boxes[p, id, 1], tab_boxes[p, id, 2], tab_boxes[p, id, 3]]
            iou=common.intersection_over_union(b1, box)
            if iou>seuil_iou and box[6]==pred_ids[p, id]:
              tp[pred_ids[p, id]]+=1
        else:
          break
              
      for box in labels2[p]:
        if not box[5]:
          break
        nbr_true_boxes[int(box[6])]+=1

      tab_nbr_reponse.append(nbr_reponse)
      tab_tp.append(tp)
      tab_true_boxes.append(nbr_true_boxes)
    if list_attributs is not None:
      pred_attr=np.array(predictions[:, :, :, :, 5+nbr_classes:])
      pred_attr=common.sigmoid(pred_attr)
      labels_attr=labels[:, :, :, :, 5+nbr_classes:]
      
      s=np.sum(masks_attr*(np.equal(np.around(pred_attr), labels_attr).astype(np.float32)), axis=(0,1, 2, 3))
      nbr=np.sum(masks_attr, axis=(0, 1, 2, 3))
      score_attr.append(s/(nbr+1E-7))
      
  tab_nbr_reponse=np.array(tab_nbr_reponse)
  tab_tp=np.array(tab_tp)
  tab_true_boxes=np.array(tab_true_boxes)
  
  precision=np.mean(tab_tp/(tab_nbr_reponse+1E-7), axis=0)
  rappel=np.mean(tab_tp/(tab_true_boxes+1E-7), axis=0)
  score=[]
  for id in range(len(precision)):
    score.append((1+beta*beta)*precision[id]*rappel[id]/(beta*beta*precision[id]+rappel[id]+1E-7))
  score=np.array(score)
  
  if list_attributs is not None:
    score_attr=np.array(score_attr)
    score_attr=np.mean(score_attr, axis=1)

  if verbose:
    print("# Score par classe:")
    for id in range(nbr_classes):
      print("  - classe '{}':".format(list_labels[id]))
      print("    - Précision: {:.2%}".format(precision[id]))
      print("    - Rappel   : {:.2%}".format(rappel[id]))
      print("    - Score    : {:.2%}".format(score[id]))
    precision=np.mean(precision)
    rappel=np.mean(rappel)
    score=np.mean(score)
    print(" -> Score final")
    print(" - Précision: {:.2%}".format(precision))
    print(" - Rappel   : {:.2%}".format(rappel))
    print(" - Score    : {:.2%}".format(score))
    if list_attributs is not None:
      print(" - Score attributs:")
      for id in range(len(list_attributs)):
        print("   - {}: {:.2%}".format(list_attributs[id], score_attr[id]))
      score_attr=np.mean(score_attr)
      print(" -> Score final: {:.2%}".format(score_attr))
    else:
      score_attr=None  
  else:
    precision=np.mean(precision)
    rappel=np.mean(rappel)
    score=np.mean(score)
    if list_attributs is not None:
      score_attr=np.mean(score_attr)
    else:
      score_attr=None  

  return score, score_attr
