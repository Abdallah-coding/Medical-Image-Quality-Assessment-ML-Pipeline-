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



---

## 2) Construction du dataset et choix méthodologiques



### Origine des données

Le dataset original provient de Kaggle. Les radiographies utilisées sont des images réelles de poumons.

Cependant, ces images sont toutes de qualité technique correcte. Afin d’entraîner un modèle capable de distinguer des images de bonne et de mauvaise qualité, il était nécessaire de disposer d’exemples dégradés.

---

### Génération artificielle des images de mauvaise qualité

Pour simuler des défauts réalistes d’acquisition, des dégradations ont été générées artificiellement à l’aide de Python (OpenCV).

Les transformations appliquées sont :

- Ajout de flou gaussien (simulation d’un mouvement du patient),
- Ajout de bruit aléatoire,
- Réduction du contraste.

Pour chaque image originale de bonne qualité, plusieurs versions dégradées ont été créées. Cela a permis de construire un jeu de données supervisé composé de deux classes : bonne qualité et mauvaise qualité.

---

### Déséquilibre du dataset

Le dataset final est déséquilibré.

Ce déséquilibre provient directement du processus de génération : pour chaque image originale, plusieurs versions dégradées ont été produites, ce qui augmente mécaniquement la proportion d’images de mauvaise qualité.

Ce point est important car il influence l’interprétation des métriques d’évaluation, notamment l’accuracy.

---

### Choix des caractéristiques (Feature Engineering)

Les caractéristiques utilisées sont :

- Netteté (variance du Laplacien),
- Contraste (écart-type des intensités),
- Luminosité moyenne,
- Entropie de Shannon,
- Approximation du rapport signal/bruit (SNR).

Ces caractéristiques ont été choisies car elles sont directement liées à la qualité technique d’une image :

- Une image floue présente moins de contours nets.
- Une image peu contrastée présente une distribution d’intensité plus concentrée.
- Une image trop sombre ou trop lumineuse peut être difficilement exploitable.
- L’entropie mesure la diversité d’information dans l’image.
- Le SNR permet d’estimer la proportion de bruit.

L’objectif était de vérifier si ces caractéristiques, définies manuellement, possèdent un pouvoir discriminant suffisant pour séparer les deux classes.

---

### Choix du modèle : régression logistique

La régression logistique a été utilisée comme modèle de référence (baseline).

Ce choix s’explique par plusieurs raisons :

- Il s’agit d’un modèle simple et interprétable.
- Il est adapté aux problèmes de classification binaire.
- Il permet d’évaluer rapidement si les caractéristiques extraites sont pertinentes.

Commencer par un modèle simple permet d’établir une base de comparaison avant d’introduire des architectures plus complexes.

---

### Limites méthodologiques

Ce projet présente plusieurs limites :

- Les dégradations sont artificielles et peuvent être plus simples à détecter que des défauts réels d’acquisition.
- Le dataset est déséquilibré.
- Les labels sont déterministes (créés par transformation).
- La régression logistique suppose une séparation linéaire entre les classes.

Ces limites doivent être prises en compte dans l’interprétation des résultats.

---

### Perspectives

Les résultats obtenus montrent que les caractéristiques extraites possèdent un pouvoir discriminant partiel, mais ne permettent pas une séparation parfaite des classes.

Dans une perspective d’évolution vers la détection de pneumonie, l’utilisation d’un modèle de type CNN apparaît pertinente, car il permet d’apprendre automatiquement des représentations spatiales complexes directement à partir des pixels.

Ce projet constitue donc une étape méthodologique préalable avant la mise en œuvre d’un modèle plus avancé.

---
