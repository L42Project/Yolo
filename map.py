import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import math

import common
import config as cfg
import model
import score

images, labels, labels2, list_labels=common.read_xmls(cfg.dir_dataset,
                                                      1,
                                                      verbose=True)
nbr_classes=len(list_labels)

images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)

dataset=tf.data.Dataset.from_tensor_slices((images, labels)).batch(cfg.batch_size)

del images, labels

images_test, labels_test, labels2_test, list_labels_test=common.read_xmls(cfg.dir_test,
                                                                          1,
                                                                          list_labels=list_labels,
                                                                          verbose=True)
images_test=np.array(images_test, dtype=np.float32)/255
labels_test=np.array(labels_test, dtype=np.float32)

test_ds=tf.data.Dataset.from_tensor_slices((images_test, labels_test)).batch(cfg.batch_size)

model=model.model(nbr_classes, cfg.nbr_boxes, cfg.cellule_y, cfg.cellule_x)
checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(cfg.dir_model))

print("Base d'appentissage:")
start=time.time()
accuracy=score.calcul_map(model, dataset, nbr_classes, labels2, verbose=True)
temps=time.time()-start

print("Base de test/validation:")
start=time.time()
accuracy=score.calcul_map(model, test_ds, nbr_classes, labels2_test, verbose=True)
temps=time.time()-start



