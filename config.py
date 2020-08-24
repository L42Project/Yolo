import numpy as np

dir_dataset="./dataset/"
dir_test=dir_dataset+"/test/"

dir_model="./my_model/"

colors=[(0, 255, 0), (0, 255, 0), (0, 255, 0)]

largeur=256
hauteur=192
cellule_x=8
cellule_y=6

r_x=int(largeur/cellule_x)
r_y=int(hauteur/cellule_y)

# Nombre maximum d'objet par image
max_objet=30
# Nombre minimum d'objet pour un entrainement
label_min_objet=500  

anchors=np.array([[3.0, 1.5], [2.0, 2.0], [1.5, 3.0]])
nbr_boxes=len(anchors)

batch_size=32
nbr_entrainement=40

# Nombre d'image générée à partir d'une image 
nbr_image_generee=10
# Nombre d'image générée à partir d'une image pour le calcul de précision 
nbr_image_generee_test=10  

# Calcul de précision sur la base d'entrainement
calcul_score=False
# Calcul de précision sur la base de test/validation
calcul_score_test=True     
# sauvegarde les valeurs des calculs de précision 
logs=True
# Nom du fichier de log
logs_file="logs.csv"       

# Pour la fonction de perte ##########
lambda_coord=5.
lambda_noobj=0.5
seuil_iou_loss=0.6

# Seuil de confidence pour les inférences
seuil_conf=0.1
