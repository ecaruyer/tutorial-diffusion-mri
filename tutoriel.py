# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: base
#     language: python
#     name: base
# ---

# %% [markdown]
# # Estimation des cartes en IRM de diffusion
# L'objectif de ce tutoriel est de se familiariser avec les données pondérées
# en diffusion et d'estimer la carte de tenseur de diffusion. À partir de cette
# carte, on peut estimer différentes mesures, comme la diffusivité moyenne ou
# l'anisotropie fractionnelle.
#
# ## Organisation des données en IRM de diffusion
# Dans la plupart des cas, les données pondérées en diffusion seront
# disponibles à la sortie de la machine au format DICOM. Étant donné que chaque
# constructeur ait des spécificités, bien qu'il s'agisse d'un standard, on
# préfère en général travailler avec un format plus simple : le format NIFTI
# (extension `.nii` ou `.nii.gz`).  Ce format permet de stocker les données
# d'images, de même que l'orientation et quelques autres méta-informations.
# Pour pouvoir interpréter les données de diffusion, on a également besoin de
# connaitre les directions et valeurs de pondération en IRM de diffusion :
# c'est généralement stocké dans deux fichiers, d'extension `.bval` et `.bvec`.
#
# ### Récupération d'un jeu de données test
# Dans le reste du tutoriel, nous allons utiliser la bibliothèque `dipy` en 
# Python. Par commodité, cette bibliothèque propose des données d'exemple.
# Commençons par charger l'un de ces jeux de données et regardons le contenu
# des fichiers `.bval` et `.bvec`.


# %%
from dipy.data import get_fnames

f_image, f_bval, f_bvec = get_fnames(name="stanford_hardi")

print("Nom des fichiers .bval et .bvec: ")
print(f_bval, f_bvec)
print()

print("Contenu du fichier ", f_bval)
print(open(f_bval).read())
print()

print("Contenu du fichier ", f_bvec)
print(open(f_bvec).read())
print()


# %% [markdown]
# ### Chargement des données d'image
# Les données d'image en elle-même sont contenues dans le fichier Nifti. On
# peut ouvrir ce fichier avec la bibliothèque Nibabel.


# %%
import nibabel as nib
img = nib.load(f_image)
print("Dimensions de l'image : ", img.shape)


# %% [markdown]
# On peut par exemple connaitre l'orientation de l'image. Celle-ci est
# représentée par une matrice de transformation affine, mais on peut demander
# à `nibabel` ce que signifie cette matrice. 


# %%
from nibabel.orientations import aff2axcodes
orientation = aff2axcodes(img.affine)

print("Matrice d'orientation : ", img.affine)
print("Orientation de l'image : ", orientation)

# %% [markdown]
# Dans cet exemple, on a une orientation `RAS`, c'est-à-dire que le premier
# axe pointe vers la droite (`R` pour *right*), le second axe pointe vers la 
# direction antérieure (`A` pour *anterior*) et le dernier axe pointe vers la
# directions supérieure (`S` pour *superior*).
#
# ### Inspection de la première image (non pondérée en diffusion)
# Si on regarde la dernière dimension de l'image, on s'aperçoit qu'on a 160 
# composantes. Cela correspond aux 160 valeurs de $b$ présentes dans le fichier
# `.bval` ouvert plus haut. On va commencer par regarder la première de 
# ces images, qui correspond à une valeur de $b=0$.


# %%
from matplotlib import pyplot as plt
first_image = img.get_fdata()[..., 0]
dim_x, dim_y, dim_z = first_image.shape

fig, axs = plt.subplots(1, 3)

axs[0].imshow(first_image[:, :, dim_z // 2])
axs[0].set_title("Vue axiale")
axs[0].axis("off")

axs[1].imshow(first_image[:, dim_z // 2, :])
axs[1].set_title("Vue sagittale")
axs[1].axis("off")

axs[2].imshow(first_image[dim_x // 2, :, :])
axs[2].set_title("Vue coronale")
axs[2].axis("off")

# %% [markdown]
# ### Représentation des directions d'encodage
# Chacune des 160 images acquises correspond à une image de référence (qui 
# correspond à $b = 0$) ou une image pondérée en diffusion (ici on a 
# $b=2000$ s/mm²). On peut regarder comment sont distribuées ces directions 
# de diffusion, on va les représenter graphiquement.

# %%
# %matplotlib widget
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

bvals, bvecs = read_bvals_bvecs(f_bval, f_bvec)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter3D(*(bvecs[bvals > 0]).T, c="gray")
ax.set_aspect("equal")
ax.axis("off");


# %% [markdown]
# ### Contraste dans les images en fonction de la direction d'encodage
# Pour se rendre compte de l'effet de la pondération en diffusion, on va ouvrir
# 3 des images qui correspondent respectivement à un encodage en `x` 
# (gauche-droite), en `y` (antero-posterieur) et en `z` (supérieur-inférieur).
# On pourrait chercher dans la table de gradients 3 directions qui 
# correspondent à ce qu'on vient de décrire ; comme il y a beaucoup de 
# directions on va essayer de faire cela de manière automatique.


# %%
import numpy as np

index_x = np.argmax(np.abs(bvecs[:, 0]))
index_y = np.argmax(np.abs(bvecs[:, 1]))
index_z = np.argmax(np.abs(bvecs[:, 2]))
print(index_x, index_y, index_z)

fig, axs = plt.subplots(1, 3)

img = img.get_fdata()

axs[0].imshow(img[:, :, dim_z // 2, index_x], cmap="gray")
axs[0].set_title("Pondération en x")
axs[0].axis("off")

axs[1].imshow(img[:, :, dim_z // 2, index_y], cmap="gray")
axs[1].set_title("Pondération en y")
axs[1].axis("off")

axs[2].imshow(img[:, :, dim_z // 2, index_z], cmap="gray")
axs[2].set_title("Pondération en z")
axs[2].axis("off");


# %% [markdown]
# ### Estimation des cartes de tenseur de diffusion
# Le tenseur est lié aux images pondérées en diffusion par une équation ;
# l'estimation du tenseur consiste à résoudre un problème d'optimisation (ou
# *fitting* en anglais). Quand on a *a minima* 6 images pondérées en diffusion
# et une image de référence, cette estimation est mathématiquement possible.
# Ici, on va utiliser la bibliothèque `dipy` pour faire cette estimation. On 
# commence par créer un masque du cerveau (pour éviter de faire une estimation 
# là où il n'y a pas de signal utile).

# %%
from dipy.segment.mask import median_otsu
_, mask = median_otsu(img, vol_idx=[0], median_radius=4, numpass=2)

fig = plt.figure()
plt.imshow(mask[:, :, dim_z // 2], cmap="gray")
plt.title("Masque du cerveau")
plt.axis("off")


# %% [markdown]
# On peut maintenant, à l'intérieur de ce masque, estimer le tenseur de
# diffusion. À partir de ce tenseur, on peut estimer la diffusivité moyen et la
# fraction d'anisotropie.

# %%
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table

gtab = gradient_table(bvals, bvecs)

# Define and fit DTI model
tensor_model = dti.TensorModel(gtab)
tensor_fit = tensor_model.fit(img, mask=mask)

# Extract FA and MD metrics
fa = tensor_fit.fa
md = tensor_fit.md


# %%
fig, axs = plt.subplots(1, 2)

m = axs[0].imshow(md[:, :, dim_z // 2], cmap="gray")
plt.colorbar(mappable=m, ax=axs[0])
axs[0].axis("off")
axs[0].set_title("Diffusivité moyenne [mm²/s]")

m = axs[1].imshow(fa[:, :, dim_z // 2], cmap="gray")
plt.colorbar(mappable=m, ax=axs[1])
axs[1].axis("off")
axs[1].set_title("Fraction d'anisotropie");

