# PhD Thesis SEYDI

```
dpmm_spatial/
│
├── data/                       # Pour stocker les données (simulées ou non)
│
├── dpmm/
│   ├── __init__.py
│   ├── sampling.py             # Fonctions pour échantillonnage
│   └── density.py              # Fonctions pour calculs de densité (f0, ftilde0, f, fbar0)
│
├── visualizations/
│   ├── __init__.py
│   ├── plot_density.py         # Visualisation heatmap, scatter, contours
│   └── plot_distL2.py          # Courbes de convergence
│
├── experiments/
│   ├── __init__.py
│   ├── compute_l2.py           # Distances L2
│   └── alpha_sweep.py          # Étude de la variation de alpha
│
├── main.py                     # Script principal
├── requirements.txt            # Dépendances du projet
└── README.md                   # Explication du projet, etc.
```



