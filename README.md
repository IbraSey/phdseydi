# PhD Thesis SEYDI


```

JdB/                                                    # Journal de bord
    ├── Fiche_Bayésien_NP.pdf
    ├── Livret_de_rapports.pdf
    └── Notes_Bayesian_modeling_earthquake.pdf
    │
papers/
    ├── BNP/
    │    ├── À lire/
    │    ├── En cours/
    │    └── Lu/
    ├── GP/
    │    ├── À lire/
    │    ├── En cours/
    │    └── Lu/
    └── Sismo/
    │    ├── À lire/
    │    ├── En cours/
    │    └── Lu/
    │
spatial_density_estimation/
    ├── brouillon_spatial.ipynb                        # Notebook faisant office de brouillon (amener à disparaître sur un temps suffisamment long)
    ├── data/                                          # Pour stocker les données (simulées ou non)
    ├── dpmm_spatial/                                  # Estimation de densité spatiale par DPMM
    │    ├── dpmm/
    │    │    ├── __init__.py
    │    │    ├── dpmm.py                              # Fonctions et classes concernant DPMM (construction, inférence, etc.) 
    │    │    └── prior_utils.py                       # Fonctions concernant la construction des priors
    │    │ 
    │    ├── experiments/
    │    │    ├── __init__.py
    │    │    └── compute_l2.py                        # Fonctions pour calcul de distances L2
    │    │ 
    │    ├── visualizations/
    │    │    ├── figures/                             # Figures illustratives concernant le DPMM
    │    │    ├── __init__.py
    │    │    └── plot.py                              # Fonctions pour visualisations (heatmap, scatter, contours, etc.)
    │    │ 
    │    └── main_dpmm.py                              # Script principal pour DPMM
    │ 
    ├── kde_spatial/                                   # Estimation de la densité spatiale par KDE
    │    ├── __init__.py
    │    ├── kde.py
    │    └── main_kde.py                               # Script principal pour KDE
    │
requirements.txt                                       # Dépendances du projet
    │
README.md                                              # Explication du projet, etc.
```


## Installation

### Prérequis

- **Python** : Version 3.12 ou supérieure
- **uv** (recommandé) ou **pip** : Pour la gestion des dépendances
- **Conda** (optionnel) : Pour créer un environnement virtuel


### Etape 0 : Si besoin, déclarer phebus comme sous-module du repo git existant : 

La présence de sous-modules est détectée par la présence du fichier caché *.gitmodules* qui contient la liste des sous-modules et de leurs caractériqstiques. 

Pour ajouter le sous-module phebus, il faut taper les instructions suivantes : 

```bash
cd phdseydi
mkdir lib_py
git submodule add --name phebus-new https://gitlab.pleiade.edf.fr/Bayesian_PSHA/phebus-new.git lib_py/phebus
```

Remarquons que le nom du sous-module ("phebus") est différent du nom du projet git distant ("phebus-new") : dans ce cas précis il est nécessaire de changer le nom car "phebus-new" contient le caractère spécial "-" ce qui empêche son utilisation par Python.

Celles-ci créeront (ou compléterons) le fichier *.gitmodules*, ainsi que d'autres fichiers de configuation.Il faut ensuite enregistrer et pousser les modifs résultantes pour que les modules soient accessibles aux autres utilisateurs.

### Etape 1 : Cloner le dépôt et tous ses sous-modules en une seule commande 

Pour cloner un depot distant avec ses sous-modules, il faut taper l'instruction suivante : 

```bash
git clone --recurse-submodules https://github.com/IbraSey/phdseydi.git
```

Il faut ensuite cloner les dépots distants correspondants aux sous-modules en tapant :

```bash
cd phdeseydi

git submodule init
git submodule update
```

### Etape 2 : Installation des autres dépendances uv

Les dépendances du projet sont listées dans le fichier *pyproject.toml*. Celui-ci est utilisé par *pixi* pour créer un environnement virtuel à la racine du dépôt (dossier *.venv*) contenant tous les paquets Python requis, grâce aux commandes suivantes :

```bash
# Installer pixi si ce n'est pas déjà fait
curl -fsSL https://pixi.sh/install.sh | sh

# Depuis la racine du projet
pixi install --locked

# Rentrer dans l'env
pixi shell

# Lancer une commande
pixi run

# Ajouter une dépendance (mets à jour le pyproject.toml)
pixi add 
```



