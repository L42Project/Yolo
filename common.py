import tensorflow as tf
from tensorflow.keras import layers, models
import xmltodict
from collections import OrderedDict
import random
import cv2
import numpy as np
from os import path
import glob
import math
import config as cfg

def sigmoid(x):
  x=np.clip(x, -50, 50)
  return 1/(1+np.exp(-x))

def softmax(x):
  e=np.exp(x)
  e_sum=np.sum(e)
  return e/e_sum

def prepare_image(image, labels, grille=True):
  img=image.copy()
  
  if grille is True:
    for x in range(cfg.r_x, cfg.largeur+cfg.r_x, cfg.r_x):
      for y in range(cfg.r_y, cfg.hauteur+cfg.r_y, cfg.r_y):
        cv2.line(img, (0, y), (x, y), (0, 0, 0), 1)
        cv2.line(img, (x, 0), (x, y), (0, 0, 0), 1)

  for y in range(cfg.cellule_y):
    for x in range(cfg.cellule_x):
      for box in range(cfg.nbr_boxes):
        if labels[y, x, box, 4]:
          ids=np.argmax(labels[y, x, box, 5:])
          x_center=int(labels[y, x, box, 0]*cfg.r_x)
          y_center=int(labels[y, x, box, 1]*cfg.r_y)
          w_2=int(labels[y, x, box, 2]*cfg.r_x/2)
          h_2=int(labels[y, x, box, 3]*cfg.r_y/2)
          x_min=x_center-w_2
          y_min=y_center-h_2
          x_max=x_center+w_2
          y_max=y_center+h_2
          cv2.rectangle(img, (x_min, y_min), (x_max, y_max), cfg.colors[ids], 1)
          cv2.circle(img, (x_center, y_center), 1, cfg.colors[ids], 2)
            
  return img

def bruit(image):
  h, w, c=image.shape
  n=np.random.randn(h, w, c)*random.randint(5, 40)
  return np.clip(image+n, 0, 255).astype(np.uint8)

def gamma(image, alpha=1.0, beta=0.0):
  return np.clip(alpha*image+beta, 0, 255).astype(np.uint8)

def intersection_over_union(boxA, boxB):
  xA=np.maximum(boxA[0], boxB[0])
  yA=np.maximum(boxA[1], boxB[1])
  xB=np.minimum(boxA[2], boxB[2])
  yB=np.minimum(boxA[3], boxB[3])
  interArea=np.maximum(0, xB-xA)*np.maximum(0, yB-yA)
  boxAArea=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
  boxBArea=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])
  return interArea/(boxAArea+boxBArea-interArea)

def infos_xmls(dir_dataset, verbose=False):
  list_labels=[]
  stats_labels=[]
  nbr_fichier=0

  if verbose:
    print("########################################################")
    print("Lecture du repertoire", dir_dataset)
  fichiers=glob.glob(dir_dataset+"/*.xml")
  if len(fichiers)==0:
    print("Le repertoire", dir_dataset, "est vide")
    quit()
  for fichier in fichiers:
    with open(fichier) as fd:
      doc=xmltodict.parse(fd.read())
      fichier=doc['annotation']['filename']
      if not path.exists(dir_dataset+fichier):
        print("Le fichier image n'existe pas ...", fichier)
        continue

      nbr_fichier+=1
      objects=doc['annotation']['object']
      if type(objects) is OrderedDict:
        objects=[objects]
      for o in objects:
        info_label=o['name']

        # Gestion du label
        lab=info_label.split(';')[0]        
        if lab=="":
          print("Probleme label ...", fichier, objects)
          quit()
        if list_labels:
          if lab in list_labels:
            stats_labels[list_labels.index(lab)]+=1
          else:
            list_labels.append(lab)
            stats_labels.append(1)
        else:
          list_labels=[lab]
          stats_labels=[1]

  if verbose:
    print("Nbr fichiers", nbr_fichier)
    print("Liste labels", list_labels)
    for i in range(len(list_labels)):
      print("   Label: {:9s} nbr: {:5d}".format(list_labels[i], stats_labels[i]))

  list_labels_final=[]
  for i in range(len(list_labels)):
    if stats_labels[i]>=cfg.label_min_objet:
      list_labels_final.append(list_labels[i])

  if verbose:
    print("   Labels retenus:", list_labels_final)
    print("########################################################")

  return list_labels, list_labels_final

def prepare_labels(fichier_image, objects, list_labels, modification=True, verbose=False):
    if verbose:
      print("Fichier:", fichier_image)
    image=cv2.imread(fichier_image)

    if image is None:
      print("Probleme avec le fichier", fichier_image)
      return None, None, None, None
    
    image_r=cv2.resize(image, (cfg.largeur, cfg.hauteur))
    
    nbr_classes=len(list_labels)
    label =np.zeros((cfg.cellule_y, cfg.cellule_x, cfg.nbr_boxes, 5+nbr_classes), dtype=np.float32)
    label2=np.zeros((cfg.max_objet, 7), dtype=np.float32)

    if modification is True:
      if not np.random.randint(4):
        image_r=cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
        image_r=np.expand_dims(image_r, axis=-1)
        image_r=np.tile(image_r, (3))
      image_r=gamma(image_r, random.uniform(0.5, 1.8), np.random.randint(60)-30)
      image_r=bruit(image_r)
      flip=np.random.randint(4)
      if flip!=3:
        image_r=cv2.flip(image_r, flip-1)
    else:
      flip=3
        
    ratio_x=cfg.largeur/image.shape[1]
    ratio_y=cfg.hauteur/image.shape[0]
    nbr_objet=0
    if type(objects) is OrderedDict:
      objects=[objects]
    for o in objects:
        info_label=o['name']
        lab=info_label.split(';')[0]
        if not lab in list_labels:
          continue
        id_class=list_labels.index(lab)
        box=o['bndbox']
          
        if flip==3:
          x_min=int(int(box['xmin'])*ratio_x)
          y_min=int(int(box['ymin'])*ratio_y)
          x_max=int(int(box['xmax'])*ratio_x)
          y_max=int(int(box['ymax'])*ratio_y)
        if flip==2:
          x_min=int((image.shape[1]-int(box['xmax']))*ratio_x)
          y_min=int(int(box['ymin'])*ratio_y)
          x_max=int((image.shape[1]-int(box['xmin']))*ratio_x)
          y_max=int(int(box['ymax'])*ratio_y)
        if flip==1:
          x_min=int(int(box['xmin'])*ratio_x)
          y_min=int((image.shape[0]-int(box['ymax']))*ratio_y)
          x_max=int(int(box['xmax'])*ratio_x)
          y_max=int((image.shape[0]-int(box['ymin']))*ratio_y)
        if flip==0:
          x_min=int((image.shape[1]-int(box['xmax']))*ratio_x)
          y_min=int((image.shape[0]-int(box['ymax']))*ratio_y)
          x_max=int((image.shape[1]-int(box['xmin']))*ratio_x)
          y_max=int((image.shape[0]-int(box['ymin']))*ratio_y)

        x_min=x_min/cfg.r_x
        y_min=y_min/cfg.r_y
        x_max=x_max/cfg.r_x
        y_max=y_max/cfg.r_y

        area=(x_max-x_min)*(y_max-y_min)
        label2[nbr_objet]=[x_min, y_min, x_max, y_max, area, 1, id_class]
        
        x_centre=int(x_min+(x_max-x_min)/2)
        y_centre=int(y_min+(y_max-y_min)/2)
        x_cell=int(x_centre)
        y_cell=int(y_centre)

        a_x_min=x_centre-cfg.anchors[:, 0]/2
        a_y_min=y_centre-cfg.anchors[:, 1]/2
        a_x_max=x_centre+cfg.anchors[:, 0]/2
        a_y_max=y_centre+cfg.anchors[:, 1]/2

        id_box=0
        best_iou=0
        for i in range(len(cfg.anchors)):
          iou=intersection_over_union([x_min, y_min, x_max, y_max], [a_x_min[i], a_y_min[i], a_x_max[i], a_y_max[i]])
          if iou>best_iou:
            best_iou=iou
            id_box=i

        label[y_cell, x_cell, id_box, 0]=(x_max+x_min)/2
        label[y_cell, x_cell, id_box, 1]=(y_max+y_min)/2
        label[y_cell, x_cell, id_box, 2]=x_max-x_min
        label[y_cell, x_cell, id_box, 3]=y_max-y_min
        label[y_cell, x_cell, id_box, 4]=1.
        label[y_cell, x_cell, id_box, 5+id_class]=1.

        #if verbose:
        #  print("   LABEL", info_label)
        #  print("   label",         label[y_cell, x_cell, id_box, 5:])

        nbr_objet=nbr_objet+1
        if nbr_objet==cfg.max_objet:
          print("Nbr objet max atteind !!!!!")
          break

    if nbr_objet:
      return image_r, label, label2
    else:
      return None, None, None

def read_xmls(dir_dataset, nbr=1, list_labels=None, verbose=False):
  if list_labels is None:
    list_all_labels, list_labels=infos_xmls(dir_dataset, verbose=verbose) # A MODIFIER ...
  
  images=[]
  labels=[]
  labels2=[]
  for fichier in glob.glob(dir_dataset+"/*.xml"):
    with open(fichier) as fd:
      doc=xmltodict.parse(fd.read())
      fichier=doc['annotation']['filename']
      if not path.exists(dir_dataset+fichier):
        print("Le fichier image n'existe pas ...", fichier)
        continue
      for i in range(nbr):
        image, label, label2=prepare_labels(dir_dataset+fichier,
                                            doc['annotation']['object'],
                                            list_labels,
                                            True if i>1 else False,
                                            verbose=False)
        if image is not None:
          images.append(image)
          labels.append(label)
          labels2.append(label2)
  images=np.array(images)
  labels=np.array(labels)
  labels2=np.array(labels2)
  return images, labels, labels2, list_labels
