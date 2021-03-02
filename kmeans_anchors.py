import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score
import cv2
import glob
import common
import config as cfg

list_all_labels, list_labels, list_labels, list_attributs=common.infos_xmls(cfg.dir_dataset, with_attribut=cfg.with_attribut, verbose=True)

images, labels, labels2, mask_attributs=common.prepare_dataset(cfg.dir_dataset,
                                                               list_labels=list_labels,
                                                               list_attributs=list_attributs,
                                                               data_augmentation=False,
                                                               verbose=True)

if len(labels2)==0:
    print("Le dataset est vide !")
    quit()

tab_box=[]
labels2=labels2.reshape(-1, 7)

for l in labels2:
    if l[4]!=0:
        tab_box.append([l[2]-l[0], l[3]-l[1]])
        
for i in range(2, 10):
    print("Nombre de anhors:", i)
    kmeans=KMeans(n_clusters=i)
    pred_y=kmeans.fit_predict(tab_box)
    print(np.ndarray.tolist(np.round(kmeans.cluster_centers_, 2)))
