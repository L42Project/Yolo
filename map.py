import tensorflow as tf
import sys
import os
import time
import cv2
import numpy as np

import common
import config as cfg
import score

list_labels=['tete', 'humain']
list_attributs=['tete:sexe:homme', 'humain:sexe:homme', 'tete:sexe:femme', 'humain:sexe:femme', 'tete:emotion:heureux']
list_attributs=None

model=common.readmodel()
if model is None:
    quit()

images, labels, labels2, mask_attributs=common.prepare_dataset(cfg.dir_dataset,
                                                               list_labels=list_labels,
                                                               list_attributs=list_attributs,
                                                               verbose=True)
nbr_attributs=0 if cfg.with_attribut is False else len(list_attributs)

images=images/255

dataset=tf.data.Dataset.from_tensor_slices((images, labels, labels2, mask_attributs)).batch(cfg.batch_size)

del images, labels

images_test, labels_test, labels2_test, mask_attributs_test=common.prepare_dataset(cfg.dir_test,
                                                                                   list_labels=list_labels,
                                                                                   list_attributs=list_attributs,
                                                                                   verbose=True)
images_test=np.array(images_test, dtype=np.float32)/255
slabels_test=np.array(labels_test, dtype=np.float32)

test_ds=tf.data.Dataset.from_tensor_slices((images_test, labels_test, labels2_test, mask_attributs_test)).batch(cfg.batch_size)

print("############# Base d'appentissage ###################")
#start=time.time()
#accuracy=score.calcul_map(model, dataset, list_labels, list_attributs, seuil_iou=cfg.seuil_iou_score, verbose=True)
#temps=time.time()-start

print("############# Base de test/validation ###############")
start=time.time()
accuracy=score.calcul_map(model, test_ds, list_labels, list_attributs, seuil_iou=cfg.seuil_iou_score, verbose=True)
temps=time.time()-start
