# Yolo

L'ensemble de ce code est expliqué et commenté dans la série "Yolo" faite sur la chaine Youtube [L42Project](https://www.youtube.com/channel/UCn09iU3hS5Fpxv0XniGv2FQ) (cf [Playlist Yolo](https://www.youtube.com/playlist?list=PLALfJegMRGp2AIqY4PcRH678fG7eCzmKr))

### Contruire un dataset

Cette version de yolo lit un dataset dont la description des images est dans des xmls (fait avec LabelImg par exemple, cf ce [tutoriel](https://www.youtube.com/watch?v=VWXXFFDqBqA)); mettre le repertoire contenant les données dans le fichier config.py

### Corriger le fichier config.py

Si nécessaire, modifier les options du fichier config.py

Utiliser kmeans_anchors.py pour avoir les tailles de boites optimales (et corriger config.py)

### Lancer l'entrainement

Lancer l'entrainement avec train.py

### Mesure de précision

Le programme d'entrainement affiche une mesure de précision tout au long de l'entrainement mais vous pouvez refaire une mesure avec la commande map.py

### Inférences

Vous pouvez faire des inférences avec :
 - inference_image.py : permet de faire des inférences sur des images en précisant un répertoire 
 - inference_camera.py : inférence sur votre cam
 - inference_youtube.py : inférence sur une vidéo youtube

### Divers

 - stats_dataset.py : donne des statistiques sur votre dataset
 