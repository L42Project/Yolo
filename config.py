import tensorflow as tf
import seaborn as sns
import numpy as np

dir_dataset=['./dataset/dataset1',
             './dataset/dataset2',
             './dataset/dataset3',
             './dataset/dataset4',
             './dataset/dataset5',
             './dataset/reportage',
             './dataset/sport',
             './dataset/Le_chemin_du_passe',
             './dataset/Amour_a_Distance']
dir_test=['./dataset/test']

train_model='my_IA'
# Généré par convert.py
fast_train_model='my_fast_IA'

my_model='model_resnet'
#my_model='model_mobilenet'
#my_model='model_vgg'

image_size=128
nbr_cellule=8

cellule_size=int(image_size/nbr_cellule)

# Nombre maximum d'objet par image
max_objet=42
# Nombre minimum d'objet pour un entrainement
label_min_objet=400  
# Taille mini d'un objet
size_mini=0

anchors=np.array([[4.75, 2.0], [7.17, 3.92], [1.81, 0.94]], dtype=np.float32)
nbr_boxes=len(anchors)

batch_size=8
nbr_entrainement=50

# Calcul de précision sur la base d'entrainement
calcul_score=False
# Calcul de précision sur la base de test/validation
calcul_score_test=True

# sauvegarde les valeurs des calculs de précision
logs=True
# Nom du fichier de log
logs_file="logs.csv"       

# Augmentation des données du dataset
aug_flipH=True
aug_flipV=True
aug_flipHV=True
aug_NB=False
aug_NEG=False
aug_R90P=True
aug_R90N=True

# Pour la fonction de perte
lambda_coord=5.
lambda_noobj=0.5
lamba_attr=1.
seuil_iou_loss=0.6

# Seuil pour calcul de score
seuil_iou_score=0.5

# Seuil de confidence pour les inférences
seuil_conf=0.3

# Sauvegarde
# Ne pas mettre de valeur trop petite, votre disque peut vite être rempli ...
# A FAIRE
save_every=10
epoch_mini=50

# OPTION EN COURS DE DEVELOPPEMENT ... NE PAS UTILISER ...
with_attribut=False

# Couleurs
colors=sns.color_palette("tab10")

