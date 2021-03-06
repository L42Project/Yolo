import tensorflow as tf

import config as cfg
import common

grid=tf.meshgrid(tf.range(cfg.nbr_cellule, dtype=tf.float32), tf.range(cfg.nbr_cellule, dtype=tf.float32))
grid=tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
grid=tf.tile(grid, (1, 1, cfg.nbr_boxes, 1))  

def yolo_loss(labels, preds, labels2, nbr_classes, mask_attr=None):
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

