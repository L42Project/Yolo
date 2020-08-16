import cv2
import numpy as np
import common
import config

images, labels, labels2, list_labels=common.read_xmls(config.dir_dataset, 5)
index=np.random.permutation(len(images))
images=images[index]
labels=labels[index]

for i in range(len(images)):
    image=common.prepare_image(images[i], labels[i], True)
    cv2.imshow("image", cv2.resize(image, (2*config.largeur, 2*config.hauteur)))
    if cv2.waitKey()&0xFF==ord('q'):
        break

