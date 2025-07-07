# PhD Thesis SEYDI

```
dpmm_spatial/
│
├── brouillon_dpmm_spatial.ipynb               # Notebook faisant office de brouillon
│
├── data/                                      # Pour stocker les données (simulées ou non)
│
├── dpmm/
│   ├── __init__.py
│   ├── sampling.py                            # Fonctions pour échantillonnage
│   └── density.py                             # Fonctions pour calculs de densité (f0, ftilde0, f, fbar)
│
├── experiments/
│   ├── __init__.py
│   ├── compute_l2.py                          # Calcul de la distance L2
│   └── alpha_sweep.py                         # Étude de la variation de alpha
│
├── kde/kde.py
│   ├── __init__.py
│   └── kde.py                                 # Estimation de la densité spatiale par KDE
│
├── main.py                                    # Script principal
├── requirements.txt                           # Dépendances du projet
│
├── visualizations/
│   ├── __init__.py
│   ├── plot_density.py                        # Visualisation heatmap, scatter, contours
│   └── plot_distL2.py                         # Courbes de convergence
│
└── README.md                                  # Explication du projet, etc.
```



