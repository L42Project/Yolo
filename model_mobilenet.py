import tensorflow as tf
from tensorflow.keras import layers, models
import config as cfg

def block_resnet(entree, filters, kernel_size, dropout=None, reduce=False):
    dropout=None
    
    result=layers.DepthwiseConv2D(kernel_size, strides=1, padding='SAME')(entree)
    result=layers.BatchNormalization()(result)
    result=layers.LeakyReLU(alpha=0.1)(result)
    if dropout is not None:
        result=layers.Dropout(dropout)(result)

    stride=1 if reduce is False else 2
    result=layers.Conv2D(filters, 1, strides=stride, padding='SAME')(result)
    result=layers.BatchNormalization()(result)
    result=layers.LeakyReLU(alpha=0.1)(result)
    if dropout is not None:
        result=layers.Dropout(dropout)(result)
    return result


def block_repeat(entree, nbr_cc, kernels, dropout=None):
    result=entree
    for kernel in kernels:
        result=block_resnet(result, nbr_cc, kernel, dropout=dropout, reduce=False)
    return result

def model(nbr_classes, nbr_attributs, nbr_boxes, nbr_cellule, nbr_cc=32):
    entree=layers.Input(shape=(cfg.image_size, cfg.image_size, 3), dtype='float32')

    result=block_repeat(entree, 1*nbr_cc, [3, 3], dropout=0.3)
    result=block_resnet(result, 1*nbr_cc, 3, dropout=0.3, reduce=True)

    result=block_repeat(result, 2*nbr_cc, [3, 3, 3], dropout=0.4)
    result=block_resnet(result, 2*nbr_cc, 3, dropout=0.4, reduce=True)

    result=block_repeat(result, 4*nbr_cc, [3, 3, 3, 3, 3], dropout=0.5)
    result=block_resnet(result, 4*nbr_cc, 3, dropout=0.5, reduce=True)
    
    result=block_repeat(result, 8*nbr_cc, [3, 3, 3], dropout=0.5)
    result=block_resnet(result, 8*nbr_cc, 3, dropout=0.5, reduce=True)

    result=layers.Conv2D(nbr_boxes*(5+nbr_classes+nbr_attributs), 1, padding='SAME')(result)
    sortie=layers.Reshape((nbr_cellule, nbr_cellule, nbr_boxes, 5+nbr_classes+nbr_attributs))(result)

    model=models.Model(inputs=entree, outputs=sortie)
    return model

