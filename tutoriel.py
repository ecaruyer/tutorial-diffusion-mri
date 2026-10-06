# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
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
%matplotlib widget
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

bvals, bvecs = read_bvals_bvecs(f_bval, f_bvec)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter3D(*(bvecs[bvals > 0]).T)
ax.set_aspect("equal")
ax.axis("off")
plt.show()
