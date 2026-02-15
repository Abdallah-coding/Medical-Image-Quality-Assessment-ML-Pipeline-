## MEDICAL IMAGE QUALITY ASSESSMENT (ML Pipeline)


## Introduction

L’objectif de ce projet est avant tout pédagogique. Il s’agit d’une étape dans mon apprentissage du Machine Learning appliqué aux images médicales.
***Note importante : ce projet ne vise pas à détecter une pathologie. Il constitue une étape intermédiaire vers un objectif plus ambitieux : développer à terme un modèle capable de détecter une pneumonie à partir d’une radiographie pulmonaire.***
Avant d’en arriver là, il m'est nécessaire de comprendre les bases :
comment construire un jeu de données supervisé,
comment extraire des caractéristiques pertinentes à partir d’images,
comment entraîner un modèle de référence (baseline),
et comment évaluer ses performances.

Dans ce projet, je me concentre donc sur l’apprentissage du processus complet d’entraînement d’un modèle.
L’objectif est d’entraîner un modèle simple à partir de caractéristiques choisies manuellement (netteté (variance du Laplacien), contraste (écart-type), luminosité moyenne, entropie de Shannon, approximation du SNR) afin de déterminer si une radiographie est de bonne ou de mauvaise qualité technique.

Cette étape me permet de comprendre la logique d’un pipeline supervisé, d’analyser le comportement des métriques d’évaluation, de justifier méthodologiquement l’utilisation future d’un modèle plus complexe comme un CNN.
Commencer par un modèle de référence permet d’évaluer si les caractéristiques extraites sont déjà discriminantes, avant d’introduire une architecture plus complexe.



Dans ce README, vous trouverez :

**1) Les résultats du modèle de référence ainsi que leur interprétation.**

**2) Les explications détaillées concernant le choix du modèle, des caractéristiques utilisées, la méthodologie suivie et les limites du dataset synthétique (images dégradées artificiellement)** 

---
---



## 1) Les résultats du modèle de référence et interprétation

Modèle utilisé : régression logistique (classification binaire), voici les résultats:


<pre> 
confusion matrix:
[[63 12]
 [12 13]]

  classification report:
              precision    recall  f1-score   support

           0      0.840     0.840     0.840        75
           1      0.520     0.520     0.520        25 

    accuracy                          0.760       100
   macro avg      0.680     0.680     0.680       100
weighted avg      0.760     0.760     0.760       100 
  
</pre>
    

***Interprétation:***

La matrice de confusion indique que:
- 63 images de mauvaise qualité ont été correctement classées
- 13 images de bonne qualité ont été correctement détectées
- 12 images de mauvaise qualité ont été à tort considérées comme bonnes
- 12 images de bonne qualité ont été à tort classées comme mauvaises

Le modèle atteint une accuracy globale de 76 %. Cependant, l’accuracy seule peut être trompeuse, car le jeu de données est déséquilibré (75 images de mauvaise qualité contre 25 images de bonne qualité).
L’analyse détaillée montre que :
Le modèle détecte correctement les images de mauvaise qualité (classe 0) avec une précision et un rappel élevés (0.84). En revanche, les performances sont plus faibles pour les images de bonne qualité (classe 1), avec un f1-score de 0.52.

Cela signifie que le modèle a plus de difficulté à identifier correctement les images de bonne qualité et en confond une partie avec des images de mauvaise qualité.
Ces résultats indiquent que les caractéristiques extraites contiennent une information discriminante, mais qu’elles ne permettent pas une séparation parfaite des classes.

Ce modèle est donc pour moi un point de référence me permettant d’évaluer l’apport futur de modèles plus complexes comme des architectures de type CNN capables de capturer des relations non linéaires dans les données.










anglais:



Introduction

This project is a learning step in my journey toward building medical image analysis models.

The long-term objective is to develop a CNN-based model capable of detecting pneumonia from chest X-ray images. However, before implementing complex architectures, it is essential to understand the fundamentals of supervised learning workflows.

In this project, I focus on building a baseline classification pipeline for technical image quality assessment. The goal is to determine whether a chest X-ray image is of good or poor quality based on handcrafted features such as sharpness, contrast, entropy, brightness, and an SNR proxy.

Starting with a baseline model allows:

validating the relevance of extracted features,

understanding evaluation metrics,

analyzing class imbalance effects,

establishing a performance reference before moving to more complex models such as CNNs.

This project emphasizes methodology and structured experimentation rather than model complexity.
