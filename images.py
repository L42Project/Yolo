import cv2
import numpy as np
import common
import config as cfg


list_all_labels, list_labels, list_labels, list_attributs=common.infos_xmls(cfg.dir_dataset, with_attribut=cfg.with_attribut, verbose=True)

if not cfg.with_attribut:
    list_attributs=None

images, labels, labels2, mask_attributs=common.prepare_dataset(cfg.dir_dataset,
                                                               list_labels=list_labels,
                                                               list_attributs=list_attributs,
                                                               data_augmentation=False,
                                                               verbose=True)

print("Nbr image:", len(images))
images=images/255

for i in range(len(images)):
    image=common.prepare_image(images[i], labels[i], True)
    cv2.imshow("image", cv2.resize(image, (2*cfg.image_size, 2*cfg.image_size)))
    key=cv2.waitKey(3)&0xFF
    if key==ord('q'):
        break

