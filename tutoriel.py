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

# %%
# Exemple de code
from dipy.data import get_fnames

f_raw, f_bval, f_bvec = get_fnames(name="stanford_hardi")
print(f_bval, f_bvec)
