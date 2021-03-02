import cv2
import tensorflow as tf
import sys
import time
import os
import numpy as np
import pafy

import common
import config as cfg

model=common.readmodel()
if model is None:
    quit()

url="https://www.youtube.com/watch?v=8EdaGlpSRCs"
video=pafy.new(url)
video_mp4=video.getbest(preftype="mp4")
cap=cv2.VideoCapture(video_mp4.url)

list_labels=['tete', 'humain']
nbr_classes=len(list_labels)
if cfg.with_attribut is True:
    list_attributs=['tete:sexe:homme', 'humain:sexe:homme', 'tete:sexe:femme', 'humain:sexe:femme', 'tete:emotion:heureux']
    nbr_attributs=len(list_attributs)
else:
    list_attributs=None
    nbr_attributs=0

while True:
  ret, frame=cap.read()
  if ret is None:
    cap.release()
    cv2.destroyAllWindows()
    quit()

  start_time=time.time()
  tab_boxes, pred_conf, ids, pred_attr, tab_index=common.inference(frame, model, nbr_classes, nbr_attributs)

  for id in tab_index:

    if ids[id]!=0:
        continue
      
    if pred_conf[id]>cfg.seuil_conf:

      y_min, x_min, y_max, x_max=tab_boxes[id]

      color=cfg.colors[ids[id]]
      color=np.array(color)*255

      cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
      cv2.rectangle(frame, (x_min, y_min), (x_max, y_min-15), color, cv2.FILLED)
      txt="{:3.0%} {}".format(pred_conf[id], list_labels[ids[id]])
      if cfg.with_attribut:
          print(">>>", pred_attr[id])
          if pred_attr[id:0]>pred_attr[id:2]:
              txt=txt+" homme"
          else:
              txt=txt+" femme"
      cv2.putText(frame, txt, (x_min, y_min-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (255, 255, 255), 1) # {%} ???

  fps=1/(time.time()-start_time)
  cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
  cv2.imshow("Frame", frame)

  key=cv2.waitKey(1)&0xFF
  if key==ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    quit()
  if key==ord('a'):
    for cpt in range(500):
      ret, frame=cap.read()
