import tensorflow as tf
from tensorflow.keras import layers, models
import config as cfg

def block(entree, nbr_cc, kernels, dropout=None):
    result=entree
    for k in kernels:
        result=layers.Conv2D(nbr_cc, k, strides=1, padding='SAME')(result)
        result=layers.BatchNormalization()(result)
        result=layers.LeakyReLU(alpha=0.1)(result)
        if dropout is not None:
            result=layers.Dropout(dropout)(result)        
    result=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')(result)
    return result

def model(nbr_classes, nbr_attributs, nbr_boxes, nbr_cellule, nbr_cc=42):
    entree=layers.Input(shape=(cfg.image_size, cfg.image_size, 3), dtype=tf.float32)

    result=block(entree, 1*nbr_cc, [3, 3], dropout=0.2)
    result=block(result, 2*nbr_cc, [3, 1, 3, 1, 3], dropout=0.3)
    result=block(result, 4*nbr_cc, [3, 1, 3, 1, 3, 1, 3, 1, 3], dropout=0.4)
    result=block(result, 8*nbr_cc, [3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3], dropout=0.5)

    result=layers.Conv2D(nbr_boxes*(5+nbr_classes), 1, padding='SAME')(result)
    sortie=layers.Reshape((nbr_cellule, nbr_cellule, nbr_boxes, 5+nbr_classes))(result)

    model=models.Model(inputs=entree, outputs=sortie)
    return model
