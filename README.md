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



