import tensorflow as tf
import sys
import time
import cv2
import os
import numpy as np

import common
import score
import loss
import config as cfg

model=__import__(cfg.my_model)

list_all_labels, list_labels, list_labels, list_attributs=common.infos_xmls(cfg.dir_dataset, with_attribut=cfg.with_attribut, verbose=True)

###### Préparation de la base d'entrainement ##############

#list_labels=['humain']

images, labels, labels2, mask_attributs=common.prepare_dataset(cfg.dir_dataset,
                                                               list_labels=list_labels,
                                                               list_attributs=None if cfg.with_attribut is False else list_attributs,
                                                               data_augmentation=True,
                                                               verbose=True)

nbr_classes=len(list_labels)
nbr_attributs=0 if cfg.with_attribut is False else len(list_attributs)
images=images/255

if len(images)==0:
    print("Dataset vide")
    quit()

print("Nombre d'image de la base d'entrainement:", len(images))
    
train_ds=tf.data.Dataset.from_tensor_slices((images, labels, labels2, mask_attributs)).shuffle(len(labels)).batch(cfg.batch_size)

del images, labels

###### Preparation de la base de test ##############

if cfg.calcul_score_test:
    images_test, labels_test, labels2_test, mask_attributs_test=common.prepare_dataset(cfg.dir_test,
                                                                                       list_labels=list_labels,
                                                                                       list_attributs=None if cfg.with_attribut is False else list_attributs,
                                                                                       data_augmentation=False,
                                                                                       verbose=True)
    images_test=images_test/255
    
    if len(images_test)==0:
        print("Dataset de test vide")
        quit()

    test_ds=tf.data.Dataset.from_tensor_slices((images_test, labels_test, labels2_test, mask_attributs_test)).batch(cfg.batch_size)

    del images_test, labels_test

####################################################

if os.path.exists(cfg.train_model):
    print("Restauration du modele", cfg.train_model)
    model=tf.keras.models.load_model(cfg.train_model)
else:
    print("Creation du modele")
    model=model.model(nbr_classes, nbr_attributs, cfg.nbr_boxes, cfg.nbr_cellule)

model.summary()

@tf.function
def train_step(images, labels, labels2, nbr_classes, mask_attr):
  with tf.GradientTape() as tape:
    predictions=model(images)
    my_loss=loss.yolo_loss(labels, predictions, labels2, nbr_classes, mask_attr)
  gradients=tape.gradient(my_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(my_loss)

def train(train_ds, nbr_entrainement):
    if cfg.logs:
        fichier_log=open(cfg.logs_file, "a")
    for entrainement in range(nbr_entrainement):
        start=time.time()
        for images, labels, labels2_, masks_attr in train_ds:
            train_step(images, labels, labels2_, nbr_classes, masks_attr)
        temps=time.time()-start
        message="Entrainement {:04d}/{:04d}: loss: {:7.4f} [temps: {:.1f} sec.]".format(entrainement+1,
                                                                                        nbr_entrainement,
                                                                                        train_loss.result(),
                                                                                        temps)
        if cfg.logs:
            message_log="{:d}:{:f}".format(entrainement+1, train_loss.result())
        
        if cfg.calcul_score:
            start=time.time()
            accuracy, accuracy_attr=score.calcul_map(model, train_ds, list_labels, list_attributs, seuil_iou=cfg.seuil_iou_score)
            temps=time.time()-start
            message+="  score={:06.2%}".format(accuracy)
            if cfg.with_attribut:
                message+="|{:06.2%}".format(accuracy_attr)
            message+=" [temps: {:.1f} sec.]".format(temps)
            if cfg.logs:
                message_log=message_log+":{:f}".format(accuracy)
        else:
            if cfg.logs:
                message_log=message_log+":"
                
        if cfg.calcul_score_test:
            start=time.time()
            accuracy_test, accuracy_test_attr=score.calcul_map(model, test_ds, list_labels, list_attributs, seuil_iou=cfg.seuil_iou_score)
            temps=time.time()-start
            message+="  score test={:06.2%}".format(accuracy_test)
            if cfg.with_attribut:
                message+="|{:06.2%}".format(accuracy_test_attr)
            message+=" [temps: {:.1f} sec.]".format(temps)
            if cfg.logs:
                message_log=message_log+":{:f}".format(accuracy_test)
        else:
            if cfg.logs:
                message_log=message_log+":"
                
        print(message)
        if cfg.logs:
            fichier_log.write(message_log+"\n")

        train_loss.reset_states()
            
    if cfg.logs:
        fichier_log.close()
            
optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)

train_loss=tf.keras.metrics.Mean()

train(train_ds, cfg.nbr_entrainement)

tf.saved_model.save(model, cfg.train_model)
