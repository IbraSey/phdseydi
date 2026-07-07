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

### Méthode 1 : Avec uv (recommandé)

```bash
# Installer uv si ce n'est pas déjà fait
curl -LsSf https://astral.sh/uv/install.sh | sh

# Depuis la racine du projet
uv sync
```

### Méthode 2 : Avec pip et un venv (conda)

```bash
# Créer un environnement conda
conda create -n stage2026 python=3.12
conda activate stage2026

# Installer le projet en mode éditable
pip install -e .
```

### Méthode 3 : Installation directe avec pip

```bash
# Depuis un environnement Python existant
pip install .
```

### Dépendances principales

Les dépendances sont définies dans `pyproject.toml` :
- openturns >= 1.27.post1
- ottoolbox (fournis localement)
- scipy >= 1.18.0
- numpy >= 2.4.6

Pour le développement :
- pytest >= 9.1.1


