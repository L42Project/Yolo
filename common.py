import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import xmltodict
from collections import OrderedDict
import random
import numpy as np
import os
import glob
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
    for x in range(cfg.cellule_size, cfg.image_size+cfg.cellule_size, cfg.cellule_size):
      for y in range(cfg.cellule_size, cfg.image_size+cfg.cellule_size, cfg.cellule_size):
        cv2.line(img, (0, y), (x, y), (0, 0, 0), 1)
        cv2.line(img, (x, 0), (x, y), (0, 0, 0), 1)

  for y in range(cfg.nbr_cellule):
    for x in range(cfg.nbr_cellule):
      for box in range(cfg.nbr_boxes):
        if labels[y, x, box, 4]:
          ids=np.argmax(labels[y, x, box, 5:])
          x_center=int(labels[y, x, box, 0]*cfg.cellule_size)
          y_center=int(labels[y, x, box, 1]*cfg.cellule_size)
          w_2=int(labels[y, x, box, 2]*cfg.cellule_size/2)
          h_2=int(labels[y, x, box, 3]*cfg.cellule_size/2)
          x_min=x_center-w_2
          y_min=y_center-h_2
          x_max=x_center+w_2
          y_max=y_center+h_2
          cv2.rectangle(img, (x_min, y_min), (x_max, y_max), cfg.colors[ids], 1)
          cv2.circle(img, (x_center, y_center), 1, cfg.colors[ids], 2)

  return img

def bruit(image):
  h, w, c=image.shape
  n=np.random.randn(h, w, c)*random.randint(5, 30)
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

def infos_xmls(dir_dataset, with_attribut=False, verbose=False):
  list_labels=[]
  stats_labels=[]
  list_attributs=[]
  stats_attributs=[]
  list_attributs_2=[]
  stats_attributs_2=[]
  nbr_fichier=0
  nbr_max_objet=0
  nbr_objet=0

  if verbose:
    print("##################################################################")
    print("Lecture du repertoire", dir_dataset)
  for dir in dir_dataset:
    fichiers=glob.glob(dir+"/*.xml")
    if len(fichiers)==0:
      print("Le repertoire", dir, "est vide")
      continue
    for fichier in fichiers:
      with open(fichier) as fd:
        doc=xmltodict.parse(fd.read())
        fichier=doc['annotation']['filename']
        if not os.path.exists(dir+'/'+fichier):
          print("Le fichier image n'existe pas ...", fichier, dir+fichier) # A CORRIGER : FICHIER ... FICHIER IMAGE
          continue

        nbr_fichier+=1
        objects=doc['annotation']['object']
        if type(objects) is OrderedDict:
          objects=[objects]
        if nbr_max_objet<len(objects):
          nbr_max_objet=len(objects)
        nbr_objet+=len(objects)
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

          # Gestion des attributs du label
          if with_attribut:
            for l in info_label.split(';')[1:]:
              nom_attr=l.split(':')[0]
              lab_att=lab+":"+nom_attr
              if list_attributs:
                if lab_att in list_attributs:
                  stats_attributs[list_attributs.index(lab_att)]+=1
                else:
                  list_attributs.append(lab_att)
                  stats_attributs.append(1)
              else:
                list_attributs=[lab_att]
                stats_attributs=[1]

              attr=l.split(':')[1]
              lab_att_2=lab_att+":"+attr
              if list_attributs_2:
                if lab_att_2 in list_attributs_2:
                  stats_attributs_2[list_attributs_2.index(lab_att_2)]+=1
                else:
                  list_attributs_2.append(lab_att_2)
                  stats_attributs_2.append(1)
              else:
                list_attributs_2=[lab_att_2]
                stats_attributs_2=[1]
  if verbose:
    print("Nombre fichiers: {: >12d}".format(nbr_fichier))
    print("Nombre maximum d'objet:{: >6d}".format(nbr_max_objet))
    print("Nombre d'objet total:{: >8d}".format(nbr_objet))
    print("Liste labels", list_labels)

    for i in range(len(list_labels)):
      print("   Label: {:.<19s}{:.>6d}".format(list_labels[i], stats_labels[i]))
      if with_attribut:
        for j in range(len(list_attributs)):
          if list_attributs[j].split(':')[0]==list_labels[i]:
            print("      {:.<23s}{:.>6d}".format(list_attributs[j].split(':')[1], stats_attributs[j]))
            for k in range(len(list_attributs_2)):
              if list_attributs_2[k].split(':')[0]==list_labels[i] and list_attributs_2[k].split(':')[1]==list_attributs[j].split(':')[1]:
                print("         - {:.<18s}{:.>6d}".format(list_attributs_2[k].split(':')[2], stats_attributs_2[k]))
    print("Labels et attribus retenus:")

  list_labels_final=[]
  list_attributs_2_final=[]
  for i in range(len(list_labels)):
    if stats_labels[i]>=cfg.label_min_objet:
      list_labels_final.append(list_labels[i])
  if with_attribut:
    for i in range(len(list_attributs_2)):
      if stats_attributs_2[i]>=cfg.label_min_objet:
        list_attributs_2_final.append(list_attributs_2[i])

  if verbose:
    print("   Labels retenus:", list_labels_final)
    if with_attribut:
      print("   Attribus retenus:", list_attributs_2_final)
    print("##################################################################")

  return list_labels, list_attributs_2, list_labels_final, list_attributs_2_final

def labels_csv(dir_dataset):
  list_labels=[]
  for dir in dir_dataset:
    fichiers=glob.glob(dir+"/*.xml")
    if len(fichiers)==0:
      print("Le repertoire", dir, "est vide")
      quit()
    for fichier in fichiers:
      with open(fichier) as fd:
        doc=xmltodict.parse(fd.read())
        objects=doc['annotation']['object']
        if type(objects) is OrderedDict:
          objects=[objects]

        for o in objects:
          info_label=o['name']
          if info_label not in list_labels:
            list_labels.append(info_label)

  return list_labels

def prepare_labels(fichier_image, objects, list_labels, list_attributs=None, flipH=False, flipV=False, NB=False, R90P=False, R90N=False, NEG=False, verbose=False):

    image=cv2.imread(fichier_image)
    if image is None:
      print("Probleme avec le fichier", fichier_image)
      return None, None, None, None

    width=image.shape[1]
    height=image.shape[0]

    image=cv2.resize(image, (cfg.image_size, cfg.image_size))
    
    ratio_x=cfg.image_size/width
    ratio_y=cfg.image_size/height
    
    nbr_classes=len(list_labels)
    nbr_attributs=0 if list_attributs is None else len(list_attributs)
    label =np.zeros((cfg.nbr_cellule, cfg.nbr_cellule, cfg.nbr_boxes, 5+nbr_classes+nbr_attributs), dtype=np.float32)
    label2=np.zeros((cfg.max_objet, 7), dtype=np.float32)

    if list_attributs is not None:
      nbr_attributs=len(list_attributs)
      mask_attributs=np.zeros((cfg.nbr_cellule, cfg.nbr_cellule, cfg.nbr_boxes, nbr_attributs), dtype=np.float32)
      list_nom_attributs=[]
      for a in list_attributs:
        list_nom_attributs.append(a.split(":")[1])
      list_nom_attributs=set(list_nom_attributs)
    else:
      mask_attributs=np.zeros((cfg.nbr_cellule, cfg.nbr_cellule, cfg.nbr_boxes, 1), dtype=np.float32)
      
    if NB is True:
      image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Normalisation
      image=image-np.min(image)
      image=image/np.max(image)*255

      image=np.expand_dims(image, axis=-1)
      image=np.tile(image, (3))

    if NEG is True:
      image=255-image

    #image_r=gamma(image_r, random.uniform(0.5, 1.8), np.random.randint(60)-30)
    #image_r=bruit(image_r)
      
    if flipV is True and flipH is True:
      image=cv2.flip(image, -1)
    if flipV is True and flipH is False:
      image=cv2.flip(image, 0)
    if flipV is False and flipH is True:
      image=cv2.flip(image, 1)

    if R90P is True:
      image=cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
      
    if R90N is True:
      image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
      
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
        x_min=int(box['xmin'])*ratio_x
        y_min=int(box['ymin'])*ratio_y
        x_max=int(box['xmax'])*ratio_x
        y_max=int(box['ymax'])*ratio_y
        if flipH is True and flipV is True:
          x_min=(width-int(box['xmax']))*ratio_x
          y_min=(height-int(box['ymax']))*ratio_y
          x_max=(width-int(box['xmin']))*ratio_x
          y_max=(height-int(box['ymin']))*ratio_y
        if flipH is True and flipV is False:
          x_min=(width-int(box['xmax']))*ratio_x
          x_max=(width-int(box['xmin']))*ratio_x
        if flipH is False and flipV is True:
          y_min=(height-int(box['ymax']))*ratio_y
          y_max=(height-int(box['ymin']))*ratio_y
        if R90P is True:
          x_min=(int(box['ymin']))*ratio_y
          y_min=(width-int(box['xmax']))*ratio_x
          x_max=(int(box['ymax']))*ratio_y
          y_max=(width-int(box['xmin']))*ratio_x
        if R90N is True:
          x_min=(height-int(box['ymax']))*ratio_y
          y_min=(int(box['xmin']))*ratio_x
          x_max=(height-int(box['ymin']))*ratio_y
          y_max=(int(box['xmax']))*ratio_x
            
        if x_max-x_min<cfg.size_mini or y_max-y_min<cfg.size_mini:
          continue

        x_min=x_min/cfg.cellule_size
        y_min=y_min/cfg.cellule_size
        x_max=x_max/cfg.cellule_size
        y_max=y_max/cfg.cellule_size
        
        area=(x_max-x_min)*(y_max-y_min)
        label2[nbr_objet]=[y_min, x_min, y_max, x_max, area, 1, id_class]

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
          iou=intersection_over_union([y_min, x_min, y_max, x_max], [a_y_min[i], a_x_min[i], a_y_max[i], a_x_max[i]])
          if iou>best_iou:
            best_iou=iou
            id_box=i

        label[y_cell, x_cell, id_box, 0]=(x_max+x_min)/2
        label[y_cell, x_cell, id_box, 1]=(y_max+y_min)/2
        label[y_cell, x_cell, id_box, 2]=x_max-x_min
        label[y_cell, x_cell, id_box, 3]=y_max-y_min
        label[y_cell, x_cell, id_box, 4]=1.
        label[y_cell, x_cell, id_box, 5+id_class]=1.

        # Gestion des attributs
        # coucou
        if list_attributs is not None:
          for attr in info_label.split(';')[1:]:
            a=attr.split(':')[0]
            if a not in list_nom_attributs:
              continue
            if lab+':'+attr in list_attributs:
              id_attr=list_attributs.index(lab+':'+attr)
              label[y_cell, x_cell, id_box, 5+nbr_classes+id_attr]=1.
            for n in range(len(list_attributs)):
              m=list_attributs[n].split(':')[1]
              #print(">>>", list_attributs[n].split(';')[0])
              if lab==list_attributs[n].split(':')[0] and m==attr.split(':')[0]:
                mask_attributs[y_cell, x_cell, id_box, n]=1.

        if verbose:
        #if True:
          print("   LABEL", info_label)
          print("      label   :", label[y_cell, x_cell, id_box, 5:5+nbr_classes])
          print("      attribut:", label[y_cell, x_cell, id_box, 5+nbr_classes:])
          print("      mask    :", mask_attributs[y_cell, x_cell, id_box])

        nbr_objet=nbr_objet+1
        if nbr_objet==cfg.max_objet:
          print("Nbr objet max atteind !!!!!")
          break

    if nbr_objet:
      return image, label, label2, mask_attributs
    else:
      return None, None, None, None

def prepare_dataset(dir_dataset, list_labels, list_attributs=None, data_augmentation=False, verbose=False):
  if verbose:
    print("Liste des labels   :", list_labels)
    if list_attributs is not None:
      print("Liste des attributs:", list_attributs)
  images=[]
  labels=[]
  labels2=[]
  masks_attributs=[]
  for dir in dir_dataset:
    for fichier in glob.glob(dir+"/*.xml"):
      with open(fichier) as fd:
        doc=xmltodict.parse(fd.read())
        fichier=doc['annotation']['filename']
        if not os.path.exists(dir+'/'+fichier):
          print("Le fichier image n'existe pas ...", fichier)
          continue

        image, label, label2, mask_attributs=prepare_labels(dir+'/'+fichier,
                                                            doc['annotation']['object'],
                                                            list_labels,
                                                            list_attributs,
                                                            verbose=False)
        if image is not None:
          images.append(image)
          labels.append(label)
          labels2.append(label2)
          masks_attributs.append(mask_attributs)
        else:
          continue

        if data_augmentation is True:
          if cfg.aug_flipH is True:
            image, label, label2, mask_attributs=prepare_labels(dir+'/'+fichier,
                                                                doc['annotation']['object'],
                                                                list_labels,
                                                                list_attributs,
                                                                flipH=True,
                                                                verbose=False)
            if image is not None:
              images.append(image)
              labels.append(label)
              labels2.append(label2)
              masks_attributs.append(mask_attributs)
            else:
              continue

          if cfg.aug_flipV is True:
            image, label, label2, mask_attributs=prepare_labels(dir+'/'+fichier,
                                                                doc['annotation']['object'],
                                                                list_labels,
                                                                list_attributs,
                                                                flipV=True,
                                                                verbose=False)
            if image is not None:
              images.append(image)
              labels.append(label)
              labels2.append(label2)
              masks_attributs.append(mask_attributs)
            else:
              continue
          
          if cfg.aug_flipHV is True:
            image, label, label2, mask_attributs=prepare_labels(dir+'/'+fichier,
                                                                doc['annotation']['object'],
                                                                list_labels,
                                                                list_attributs,
                                                                flipH=True, flipV=True,
                                                                verbose=False)
            if image is not None:
              images.append(image)
              labels.append(label)
              labels2.append(label2)
              masks_attributs.append(mask_attributs)
            else:
              continue

          if cfg.aug_R90P is True:
            image, label, label2, mask_attributs=prepare_labels(dir+'/'+fichier,
                                                                doc['annotation']['object'],
                                                                list_labels,
                                                                list_attributs,
                                                                R90P=True,
                                                                verbose=False)
            if image is not None:
              images.append(image)
              labels.append(label)
              labels2.append(label2)
              masks_attributs.append(mask_attributs)
            else:
              continue
            
          if cfg.aug_R90N is True:
            image, label, label2, mask_attributs=prepare_labels(dir+'/'+fichier,
                                                                doc['annotation']['object'],
                                                                list_labels,
                                                                list_attributs,
                                                                R90N=True,
                                                                verbose=False)
            if image is not None:
              images.append(image)
              labels.append(label)
              labels2.append(label2)
              masks_attributs.append(mask_attributs)
            else:
              continue
          
          if cfg.aug_NB is True:
            image, label, label2, mask_attributs=prepare_labels(dir+'/'+fichier,
                                                                doc['annotation']['object'],
                                                                list_labels,
                                                                list_attributs,
                                                                NB=True,
                                                                verbose=False)
            if image is not None:
              images.append(image)
              labels.append(label)
              labels2.append(label2)
              masks_attributs.append(mask_attributs)
            else:
              continue

          if cfg.aug_NEG is True:
            image, label, label2, mask_attributs=prepare_labels(dir+'/'+fichier,
                                                                doc['annotation']['object'],
                                                                list_labels,
                                                                list_attributs,
                                                                NEG=True,
                                                                verbose=False)
            if image is not None:
              images.append(image)
              labels.append(label)
              labels2.append(label2)
              masks_attributs.append(mask_attributs)
            else:
              continue

            
  images=np.array(images, dtype=np.float32)
  labels=np.array(labels, dtype=np.float32)
  labels2=np.array(labels2, dtype=np.float32)
  masks_attributs=np.array(masks_attributs, dtype=np.float32)
  return images, labels, labels2, masks_attributs

def readmodel(dir_file=None):
  if dir_file is not None:
    if os.path.exists(dir_file):
      print("Lecture du modele:", dir_file)
      model=tf.saved_model.load(dir_file)
      return model
    else:
      return None  
  if os.path.exists(cfg.fast_train_model):
    print("Lecture du modele:", cfg.fast_train_model)
    model=tf.saved_model.load(cfg.fast_train_model)
    return model
  if os.path.exists(cfg.train_model):
    print("Lecture du modele:", cfg.train_model)
    model=tf.saved_model.load(cfg.train_model)
    return model
  print("Aucun modele à lire.")
  return None
    
def inference(image_originale, model, nbr_classes, nbr_attributs):
  ratio_x=image_originale.shape[1]/cfg.image_size
  ratio_y=image_originale.shape[0]/cfg.image_size

  image=cv2.resize(image_originale, (cfg.image_size, cfg.image_size))/255

  grid=np.meshgrid(np.arange(cfg.nbr_cellule, dtype=np.float32), np.arange(cfg.nbr_cellule, dtype=np.float32))
  grid=np.expand_dims(np.stack(grid, axis=-1), axis=2)
  grid=np.tile(grid, (1, 1, cfg.nbr_boxes, 1))

  if False:
    infer=model.signatures['serving_default']
    output_tensorname=list(infer.structured_outputs.keys())[0]
    predictions=infer(tf.convert_to_tensor(image, dtype=tf.float32, name="input_1"))[output_tensorname]
  else:  
    predictions=model(np.array([image], dtype=np.float32))
  pred_boxes=predictions[0, :, :, :, 0:4]
  pred_conf=sigmoid(predictions[0, :, :, :, 4])
  pred_classes=softmax(predictions[0, :, :, :, 5:5+nbr_classes])
  if cfg.with_attribut:
    pred_attr=softmax(predictions[0, :, :, :, 5+nbr_classes:])
  ids=np.argmax(pred_classes, axis=-1)

  x_center=((grid[:, :, :, 0]+sigmoid(pred_boxes[:, :, :, 0]))*cfg.cellule_size)
  y_center=((grid[:, :, :, 1]+sigmoid(pred_boxes[:, :, :, 1]))*cfg.cellule_size)
  w=(np.exp(pred_boxes[:, :, :, 2])*cfg.anchors[:, 0]*cfg.cellule_size)
  h=(np.exp(pred_boxes[:, :, :, 3])*cfg.anchors[:, 1]*cfg.cellule_size)

  x_min=(x_center-w/2)*ratio_x
  y_min=(y_center-h/2)*ratio_y
  x_max=(x_center+w/2)*ratio_x
  y_max=(y_center+h/2)*ratio_y

  tab_boxes=np.stack([y_min, x_min, y_max, x_max], axis=-1).reshape(-1, 4).astype(np.float32)
  pred_conf=pred_conf.reshape(-1)
  ids=ids.reshape(-1)
  if cfg.with_attribut:
    pred_attr=pred_attr.reshape(-1, nbr_attributs)
  else:
    pred_attr=None
  tab_index=tf.image.non_max_suppression(tab_boxes, pred_conf, 42)

  return tab_boxes.astype(np.int32), pred_conf, ids, pred_attr, tab_index

