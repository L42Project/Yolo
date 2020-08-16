import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import common
import config as cfg
import model
import score

#import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'

###### Préparation de la base d'entrainement ##############

images, labels, labels2, list_labels=common.read_xmls(cfg.dir_dataset,
                                                      cfg.nbr_image_generee,
                                                      verbose=True)
nbr_classes=len(list_labels)
images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)

index=np.random.permutation(len(images))
images=images[index]
labels=labels[index]
labels2=labels2[index]

train_ds=tf.data.Dataset.from_tensor_slices((images, labels)).batch(cfg.batch_size)

del images, labels

####################################################

###### Preparation de la base de test ##############

images_test, labels_test, labels2_test, list_labels_test=common.read_xmls(cfg.dir_test,
                                                                          cfg.nbr_image_generee_test,
                                                                          list_labels=list_labels,
                                                                          verbose=True)
images_test=np.array(images_test, dtype=np.float32)/255
labels_test=np.array(labels_test, dtype=np.float32)

test_ds=tf.data.Dataset.from_tensor_slices((images_test, labels_test)).batch(cfg.batch_size)

del images_test, labels_test

####################################################

def my_loss(labels, preds, nbr_classes):
    grid=tf.meshgrid(tf.range(cfg.cellule_x, dtype=tf.float32), tf.range(cfg.cellule_y, dtype=tf.float32))
    grid=tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    grid=tf.tile(grid, (1, 1, cfg.nbr_boxes, 1))
    
    preds_xy    =tf.math.sigmoid(preds[:, :, :, :, 0:2])+grid
    preds_wh    =preds[:, :, :, :, 2:4]
    preds_conf  =tf.math.sigmoid(preds[:, :, :, :, 4])
    preds_classe=tf.math.sigmoid(preds[:, :, :, :, 5:5+nbr_classes])

    preds_wh_half=preds_wh/2
    preds_xymin=preds_xy-preds_wh_half
    preds_xymax=preds_xy+preds_wh_half
    preds_areas=preds_wh[:, :, :, :, 0]*preds_wh[:, :, :, :, 1]

    l2_xy_min=labels2[:, :, 0:2]
    l2_xy_max=labels2[:, :, 2:4]
    l2_area  =labels2[:, :, 4]
    
    preds_xymin=tf.expand_dims(preds_xymin, 4)
    preds_xymax=tf.expand_dims(preds_xymax, 4)
    preds_areas=tf.expand_dims(preds_areas, 4)

    labels_xy=labels[:, :, :, :, 0:2]
    labels_wh=tf.math.log(labels[:, :, :, :, 2:4]/cfg.anchors)
    labels_wh=tf.where(tf.math.is_inf(labels_wh), tf.zeros_like(labels_wh), labels_wh)
    
    conf_mask_obj=labels[:, :, :, :, 4]
    labels_classe=labels[:, :, :, :, 5:5+nbr_classes]

    conf_mask_noobj=[]
    for i in range(len(preds)):
        xy_min=tf.maximum(preds_xymin[i], l2_xy_min[i])
        xy_max=tf.minimum(preds_xymax[i], l2_xy_max[i])
        intersect_wh=tf.maximum(xy_max-xy_min, 0.)
        intersect_areas=intersect_wh[..., 0]*intersect_wh[..., 1]
        union_areas=preds_areas[i]+l2_area[i]-intersect_areas
        ious=tf.truediv(intersect_areas, union_areas)
        best_ious=tf.reduce_max(ious, axis=3)
        conf_mask_noobj.append(tf.cast(best_ious<cfg.seuil_iou_loss, tf.float32)*(1-conf_mask_obj[i]))
    conf_mask_noobj=tf.stack(conf_mask_noobj)

    preds_x=preds_xy[..., 0]
    preds_y=preds_xy[..., 1]
    preds_w=preds_wh[..., 0]
    preds_h=preds_wh[..., 1]
    labels_x=labels_xy[..., 0]
    labels_y=labels_xy[..., 1]
    labels_w=labels_wh[..., 0]
    labels_h=labels_wh[..., 1]

    loss_xy=tf.reduce_sum(conf_mask_obj*(tf.math.square(preds_x-labels_x)+tf.math.square(preds_y-labels_y)), axis=(1, 2, 3))
    loss_wh=tf.reduce_sum(conf_mask_obj*(tf.math.square(preds_w-labels_w)+tf.math.square(preds_h-labels_h)), axis=(1, 2, 3))

    loss_conf_obj=tf.reduce_sum(conf_mask_obj*tf.math.square(preds_conf-conf_mask_obj), axis=(1, 2, 3))
    loss_conf_noobj=tf.reduce_sum(conf_mask_noobj*tf.math.square(preds_conf-conf_mask_obj), axis=(1, 2, 3))

    loss_classe=tf.reduce_sum(tf.math.square(preds_classe-labels_classe), axis=4)
    loss_classe=tf.reduce_sum(conf_mask_obj*loss_classe, axis=(1, 2, 3))

    loss=cfg.lambda_coord*loss_xy+cfg.lambda_coord*loss_wh+loss_conf_obj+cfg.lambda_noobj*loss_conf_noobj+loss_classe
    return loss

model=model.model(nbr_classes, cfg.nbr_boxes, cfg.cellule_y, cfg.cellule_x)

@tf.function
def train_step(images, labels, nbr_classes):
  with tf.GradientTape() as tape:
    predictions=model(images)
    loss=my_loss(labels, predictions, nbr_classes)
  gradients=tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)

def train(train_ds, nbr_entrainement):
    if cfg.logs:
        fichier_log=open(cfg.logs_file, "a")
    for entrainement in range(nbr_entrainement):
        start=time.time()
        for images, labels in train_ds:
            train_step(images, labels, nbr_classes)
        temps=time.time()-start
        message="Entrainement {:04d}/{:04d}: loss: {:7.4f} [temps: {:.1f} sec.]".format(entrainement+1,
                                                                                        nbr_entrainement,
                                                                                        train_loss.result(),
                                                                                        temps)
        if cfg.logs:
            message_log="{:d}:{:f}".format(entrainement, train_loss.result())
        
        if cfg.calcul_score:
            start=time.time()
            accuracy=score.calcul_map(model, train_ds, nbr_classes, labels2)
            temps=time.time()-start
            message=message+"  score={:.2%} [temps: {:.1f} sec.]".format(accuracy, temps)
            if cfg.logs:
                message_log=message_log+":{:f}".format(accuracy)
        else:
            if cfg.logs:
                message_log=message_log+":"
                
        if cfg.calcul_score_test:
            start=time.time()
            accuracy_test=score.calcul_map(model, test_ds, nbr_classes, labels2_test)
            temps=time.time()-start
            message=message+"  score test={:.2%} [temps: {:.1f} sec.]".format(accuracy_test, temps)
            if cfg.logs:
                message_log=message_log+":{:f}".format(accuracy_test)
        else:
            if cfg.logs:
                message_log=message_log+":"
                
        print(message)
        if cfg.logs:
            fichier_log.write(message_log+"\n")
        if entrainement and not entrainement%20:
            checkpoint.save(file_prefix=cfg.dir_model+"/model")
    if cfg.logs:
        fichier_log.close()

            
optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)
train_loss=tf.keras.metrics.Mean()

checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(cfg.dir_model))

train(train_ds, cfg.nbr_entrainement)
checkpoint.save(file_prefix=cfg.dir_model+"/model")
