# Utiliser l'API SPIN-H

Ce guide explique comment utiliser le package sans connaitre en detail
l'implementation des processus gaussiens, du modele ETAS ou des samplers de Gibbs.

L'usage recommande suit normalement toujours le meme schema :

1. Preparer les domaines spatiaux ;
2. Construire un `EventCatalog` ;
3. Choisir `SSGCModel` ou `SPINHModel` ;
4. Configurer et lancer l'inference ;
5. Analyser l'objet `GibbsResults` retourne.

L'utilisateur travaille principalement avec le modèle et l'objet `GibbsResults`.
Chaque modèle expose directement sa méthode `model.gibbs(...)`. Les samplers
sont des moteurs internes et ne sont pas nécessaires pour un usage standard.

## Installation

Depuis la racine du depot :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

L'installation editable permet d'importer `package` depuis les scripts sans
modifier manuellement `sys.path`.

## Choisir le modele

### SSGC

Utiliser `SSGCModel` lorsque les evenements sont decrits uniquement par une
intensite spatiale de fond :

\[
\mu(s) = \widetilde{\mu}(s)\,\sigma(f(s)).
\]

- `\widetilde{\mu}` est constante par domaine spatial ;
- `f` est un champ gaussien latent ;
- il n'y a pas de relation parent-enfant entre les evenements.

### SPIN-H

Utiliser `SPINHModel` lorsque l'intensite contient egalement une composante de
declenchement de type Hawkes/ETAS :

\[
\lambda(t,s) = \mu(s) + \sum_{t_j<t}\phi(t-t_j,s-s_j,m_j).
\]

Chaque evenement peut alors etre :

- un evenement de fond ;
- un evenement declenche par un evenement anterieur.

Le modele peut etre marqué par les magnitudes. Il est marqué des que `alpha` ou
`gamma` est renseigne dans `ETASParameters`.

## Importations recommandees

```python
import numpy as np

from package import (
    ETASParameters,
    EventCatalog,
    GPParameters,
    SPINHGibbsConfig,
    GibbsConfig,
    SPINHModel,
    SSGCModel,
    generate_voronoi_cells,
    simulate_hawkes_process,
    simulate_spatial_process,
)
```

## Definir les domaines spatiaux

Les domaines sont toujours des polygones Shapely non vides, valides et sans
recouvrement interieur.

### Rectangles construits manuellement

```python
from shapely.geometry import box

polygons = [
    box(0.0, 0.0, 1.0, 1.0),
    box(1.0, 0.0, 2.0, 1.0),
]
```

### Pavage de Voronoi borne

```python
polygons, germs = generate_voronoi_cells(
    n_germs=6,
    X_bounds=(0.0, 2.0),
    Y_bounds=(0.0, 2.0),
    rng_seed=42,
)
```

### Objet `DomainPartition`

La plupart des utilisateurs peuvent appeler `Model.from_polygons(...)`
directement. Pour manipuler explicitement le pavage :

```python
from package import DomainPartition

domains = DomainPartition.from_polygons(
    polygons,
    initial_log_intensities=0.0,
)
```

Methodes utiles :

- `domains.polygons` : polygones Shapely ;
- `domains.areas` : surfaces ;
- `domains.centroids` : centres geometriques ;
- `domains.locate(x, y)` : indice du domaine de chaque point, ou `-1` ;
- `domains.validate_points(x, y)` : leve une erreur si un point est hors domaine.

## Construire un catalogue

```python
catalog = EventCatalog(
    t=np.array([0.2, 0.7, 1.4, 2.1]),
    x=np.array([0.2, 0.3, 1.2, 1.5]),
    y=np.array([0.2, 0.3, 0.4, 0.6]),
    magnitudes=np.array([2.1, 2.5, 2.2, 2.8]),
)
```

Contraintes :

- `t`, `x` et `y` doivent avoir la meme longueur ;
- les temps doivent etre tries par ordre croissant ;
- tous les points doivent appartenir a un domaine du modele ;
- les temps doivent etre inferieurs ou egaux a la duree d'observation ;
- un SPIN-H marque exige une magnitude par evenement.

Pour un SSGC ou un SPIN-H non marque, `magnitudes` peut etre omis.

## Simuler des donnees SSGC

```python
simulation = simulate_spatial_process(
    X_bounds=(0.0, 2.0),
    Y_bounds=(0.0, 2.0),
    T=20.0,
    polygons=polygons,
    mus=[4.0, 6.0, 3.0, 5.0, 7.0, 4.0],
    rng_seed=42,
)

catalog = simulation.catalog
```

`mus` contient une intensite de base positive par domaine. Une seule valeur peut
etre fournie pour utiliser la meme intensite partout.

Un champ latent personnalise accepte des tableaux `x`, `y` et retourne un
tableau de meme forme :

```python
def latent_field(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)

simulation = simulate_spatial_process(
    polygons=polygons,
    mus=5.0,
    f=latent_field,
    T=20.0,
    rng_seed=42,
)
```

Objets utiles dans `SpatialProcessSimulation` :

- `simulation.catalog` ;
- `simulation.domains` ;
- `simulation.baseline_intensities` ;
- `simulation.grid` ;
- `simulation.spatial_components(x, y)`.

## Simuler des donnees SPIN-H

```python
true_etas = ETASParameters(
    A=0.5,
    alpha=0.8,
    c=0.02,
    p=1.3,
    d=0.05,
    q=1.8,
    gamma=0.5,
)

simulation = simulate_hawkes_process(
    X_bounds=(0.0, 2.0),
    Y_bounds=(0.0, 2.0),
    T=20.0,
    polygons=polygons,
    mus=[4.0, 6.0, 3.0, 5.0, 7.0, 4.0],
    f=latent_field,
    etas_parameters=true_etas,
    beta=2.3,
    magnitude_min=2.0,
    magnitude_max=6.0,
    rng_seed=42,
)

catalog = simulation.catalog
```

La simulation conserve la verite terrain :

- `simulation.parent_indices` : `-1` pour le fond, sinon indice du parent ;
- `simulation.branching_labels` : `0` pour le fond, sinon `parent + 1` ;
- `simulation.generations` : generation de chaque evenement ;
- `simulation.is_background` : masque booleen ;
- `simulation.n_background` et `simulation.n_triggered`.

## Ajuster un modele SSGC

```python
model = SSGCModel.from_polygons(
    polygons=polygons,
    duration=20.0,
    x_bounds=(0.0, 2.0),
    y_bounds=(0.0, 2.0),
    gp_prior=GPParameters(variance=1.0, length_scale=0.3),
    eps_prior_variance=1.0,
    eps_prior_length_scale=0.2,
    nu_prior_rate=0.5,
    jitter=1e-5,
)

fit = model.gibbs(
    catalog,
    config=GibbsConfig(
        n_iter=3000,
        thin=2,
        mala_step=0.15,
        use_calibration=True,
        verbose=True,
        verbose_every=300,
    ),
    gp_backend="exact",
    rng_seed=42,
)
```

`fit` est un objet `GibbsResults`. Il contient les chaines, les diagnostics et
les methodes d'analyse posterior.

## Ajuster un modele SPIN-H

Les valeurs dans `ETASParameters` servent d'initialisation du Gibbs. Elles ne
sont pas considerees comme connues pendant l'inference.

```python
etas_init = ETASParameters(
    A=0.4,
    alpha=0.8,
    c=0.02,
    p=1.3,
    d=0.05,
    q=1.8,
    gamma=0.5,
)

model = SPINHModel.from_polygons(
    polygons=polygons,
    duration=20.0,
    x_bounds=(0.0, 2.0),
    y_bounds=(0.0, 2.0),
    gp_prior=GPParameters(variance=1.0, length_scale=0.3),
    eps_prior_variance=1.0,
    eps_prior_length_scale=0.2,
    nu_prior_rate=0.5,
    jitter=1e-5,
    etas_parameters=etas_init,
    magnitude_min=2.0,
    magnitude_max=6.0,
)

fit = model.gibbs(
    catalog,
    config=SPINHGibbsConfig(
        n_iter=6000,
        thin=2,
        mala_step=0.14,
        use_calibration=True,
        verbose=True,
        verbose_every=600,
        learn_beta=True,
        beta_init=2.0,
        sigma_mh_etas=0.05,
        sigma_mh_beta=0.1,
        adaptation_start=200,
    ),
    gp_backend="sparse",
    rng_seed=42,
)
```

Pour un SPIN-H non marque :

```python
etas_init = ETASParameters(A=0.4, c=0.02, p=1.3, d=0.05, q=1.8)
```

Dans ce cas, `alpha` et `gamma` sont absents et les magnitudes ne sont pas
obligatoires.

## Choisir le backend du GP

### `gp_backend="exact"`

- representation exacte aux positions observees ;
- choix simple pour les petits et moyens catalogues ;
- cout memoire et calculatoire plus eleve lorsque le catalogue grandit ;
- compatible avec `learn_nu=True`.

### `gp_backend="sparse"`

- approximation du champ par une base de Fourier ;
- souvent preferable pour des catalogues plus grands ;
- `learn_nu=True` n'est actuellement pas supporte ;
- les hyperparametres de `GPParameters` determinent la base utilisee.

Le backend se choisit lors de l'appel au modèle :

```python
fit = model.gibbs(catalog, gp_backend="sparse")  # ou "exact"
```

Une base Fourier personnalisée peut être transmise avec
`model.gibbs(..., gp_backend="sparse", sparse_gp=base)`.

## Configurer le MCMC

### `GibbsConfig` et `SPINHGibbsConfig`

Les deux configurations partagent les paramètres principaux du Gibbs :

| Parametre | Role |
|---|---|
| `n_iter` | Nombre total d'iterations du Gibbs |
| `thin` | Stocke une iteration toutes les `thin` iterations |
| `mala_step` | Pas MALA des log-intensites par domaine |
| `use_calibration` | Calibre les hyperparametres GP avant le Gibbs |
| `calibration_method` | `"openturns"` par défaut, ou `"sklearn"` |
| `calibration_target` | `"homogeneous"` par défaut ; `"zone_corrected"` conserve l’ancienne cible par domaines |
| `learn_nu` | Met a jour variance et longueur du GP |
| `verbose` | Affiche la progression |
| `verbose_every` | Frequence des messages |
| `compute_emu` | Calcule l'erreur integree si une intensite vraie est fournie |

Le burn-in n'est pas fixe dans la configuration. Il est choisi lors de l'analyse :

```python
summary = fit.summary(burn_in=0.5)
```

Eviter un thinning agressif : il reduit le stockage mais n'ameliore pas le
melange de la chaine. `thin=1` ou `thin=2` est un bon point de depart.

`SPINHGibbsConfig` ajoute les paramètres propres aux mises à jour ETAS :

| Parametre | Role |
|---|---|
| `learn_beta` | Met a jour le parametre Gutenberg-Richter `beta` |
| `beta_init` | Valeur initiale de `beta` |
| `beta_prior` | Hyperparametres Gamma `a_beta`, `b_beta` |
| `theta_priors` | Hyperparametres Gamma des parametres ETAS |
| `sigma_mh_etas` | Echelle initiale des propositions ETAS |
| `sigma_mh_beta` | Echelle initiale de la proposition de `beta` |
| `adaptation_start` | Debut de l'Adaptive Metropolis |
| `proposal_jitter` | Regularisation numerique des covariances adaptatives |

Exemple de priors ETAS informatifs :

```python
theta_priors = {
    "a_A": 5.0, "b_A": 10.0,
    "a_alpha": 8.0, "b_alpha": 10.0,
    "a_c": 2.0, "b_c": 100.0,
    "a_p": 4.0, "b_p": 10.0,
    "a_d": 2.0, "b_d": 40.0,
    "a_q": 9.0, "b_q": 10.0,
    "a_gamma": 5.0, "b_gamma": 10.0,
}
```

## Exploiter `GibbsResults`

### Resume posterior

```python
summary = fit.summary(burn_in=0.5)
```

Pour SSGC, les cles principales sont :

- `eps_hat` : log-intensite moyenne par domaine ;
- `f_data_hat` : champ latent moyen aux evenements ;
- `nu_hat` : variance et longueur de correlation GP.

Pour SPIN-H s'ajoutent :

- `theta_phi_hat` : parametres ETAS moyens ;
- `p_background` : probabilite de fond par evenement ;
- `beta_hat` si `beta` est appris.

### Taux d'acceptation

```python
print(fit.acceptance_rates)
```

Les taux sont des diagnostics, pas des criteres suffisants de convergence. Il
faut egalement inspecter les traces, l'autocorrelation et, idealement, plusieurs
chaines independantes.

### Acces aux chaines

```python
eps_chain = fit.eps_chain
latent_counts = fit.latent_point_counts
etas_chain = fit.etas_chain          # None pour SSGC
branching_chain = fit.branching_chain  # None pour SSGC
```

L'objet se comporte aussi comme un dictionnaire en lecture :

```python
raw_theta_chain = fit["theta_phi"]
```

`fit.raw` est disponible pour les analyses avancees, mais ne constitue pas
l'interface recommandee pour les operations courantes.

### Intensite de fond posterior

Disponible pour SSGC et SPIN-H :

```python
x_eval = np.linspace(0.0, 2.0, 50)
y_eval = np.full(50, 1.0)

mu = fit.background_intensity(
    x=x_eval,
    y=y_eval,
    burn_in=0.5,
)
```

### Intensite conditionnelle SPIN-H

```python
t_eval = np.full(50, 20.0)

mu, triggering, total = fit.conditional_intensity(
    t=t_eval,
    x=x_eval,
    y=y_eval,
    burn_in=0.5,
)
```

- `mu` : composante de fond ;
- `triggering` : somme des contributions des evenements anterieurs ;
- `total` : `mu + triggering`.

### Intensites SPIN-H en snapshots

Pour visualiser les intensites SPIN-H a plusieurs temps, utilise les snapshots statiques :

```python
snapshots = fit.plot_conditional_intensity_snapshots(
    times=np.linspace(0.2 * model.duration, model.duration, 4),
    burn_in=0.5,
    nx=50,
    ny=50,
    cmap_background="viridis",
    cmap_triggering="magma",
    cmap_total="inferno",
)
```

Le panel de gauche affiche l'intensite background spatiale posterior moyenne.
Les autres panels affichent l'intensite triggered et l'intensite totale a
plusieurs temps.

### Traces et histogrammes

```python
fit.plot_traces(
    burn_in=0.5,
    hist_color="steelblue",
    burn_in_color="red",
)
```

Les traces affichent toute la chaine stockee. Les histogrammes utilisent
uniquement les tirages post burn-in. La ligne verticale marque la limite du
burn-in.

### Declustering SPIN-H sans verite connue

```python
diagnostics = fit.plot_declustering(
    burn_in=0.5,
    background_threshold=0.5,
)
```

Le declustering est realise en deux etapes :

1. background si `p_background >= background_threshold` ;
2. sinon, choix du parent modal parmi les labels de parent strictement positifs.

Le premier panel montre les probabilites background. Le second montre l'arbre
de branchement ; les noeuds sont colores par magnitude et les liens sont noirs.

### Declustering avec verite connue

Pour des donnees issues de `simulate_hawkes_process` :

```python
diagnostics = fit.plot_declustering(
    burn_in=0.5,
    background_threshold=0.5,
    true_parent=simulation.branching_labels,
)
```

La fonction affiche alors un `classification_report` background/triggered.
Elle accepte aussi directement `simulation.parent_indices` ; la convention est
detectee automatiquement.

Quelques cles retournees :

- `p_bg` ;
- `predicted_background` ;
- `parent_mode` ;
- `parent_probability` : probabilite du parent conditionnelle a triggered ;
- `generation` ;
- `parent_accuracy_triggered` lorsque la verite est fournie ;
- `classification_report` lorsque la verite est fournie.

## Visualiser les simulations

```python
from visualization import plot_process_dashboard, plot_voronoi_cells

plot_voronoi_cells(
    polygons,
    germs,
    X_bounds=(0.0, 2.0),
    Y_bounds=(0.0, 2.0),
)

plot_process_dashboard(
    simulation.background_simulation,  # ou une SpatialProcessSimulation
    cmap="viridis",
    latent_cmap="coolwarm",
)
```

## Sauvegarder les figures

Les fonctions de visualisation acceptent generalement :

```python
savefigure=True
title_savefig="spinh/declustering"
```

Par defaut, les figures sont sauvegardees sous :

```text
package/figures/
```

Une extension `.pdf` est ajoutee si aucune extension n'est fournie. Les figures vectorielles sont sauvegardees en PDF. Les figures raster sont sauvegardees en PNG a `600` dpi.

## Utiliser une intensite vraie pendant une simulation

Le diagnostic `E_mu` est optionnel. Il faut activer `compute_emu` et fournir la
fonction de reference au moment du fit :

```python
config = GibbsConfig(
    n_iter=3000,
    compute_emu=True,
    emu_every=10,
)

fit = model.gibbs(
    catalog,
    config=config,
    rng_seed=42,
    reference_intensity=lambda x, y: simulation.spatial_components(x, y)[3],
)
```

Ce diagnostic ne doit pas etre utilise avec des donnees reelles lorsque
l'intensite vraie est inconnue.

## Methodes du modele utiles sans inference

### SSGC

```python
baseline = model.baseline_intensity(x, y, eps)
mu = model.background_intensity(x, y, eps, latent_gp)
covariance = model.epsilon_prior_covariance()
```

### SPIN-H

```python
triggering = model.triggering_intensity(t, x, y, catalog, parameters)
mu, triggering, total = model.conditional_intensity(
    t, x, y, catalog, eps, latent_gp, parameters
)
temporal_mass = model.temporal_compensator(catalog.t, parameters)
total_triggering_mass = model.triggering_compensator(catalog, parameters)
```

Ces methodes evaluent le modele pour des parametres fournis. Elles ne lancent
pas d'inference.

## Moteurs Gibbs internes

`SSGC_GibbsSampler` et `SPIN_H_GibbsSampler` implémentent les mises à jour MCMC
utilisées par les modèles. Ils ne constituent pas une seconde API publique :
utiliser `model.gibbs(...)` garantit la validation du catalogue, la bonne
configuration du backend GP et la construction cohérente de `GibbsResults`.

## Conseils de diagnostic

1. Commencer avec un petit nombre d'iterations pour verifier le code.
2. Augmenter ensuite `n_iter` et conserver `thin` faible.
3. Choisir le burn-in apres inspection des traces.
4. Verifier que les traces post burn-in paraissent stationnaires.
5. Examiner `fit.acceptance_rates`.
6. Pour SPIN-H, comparer la proportion background estimee a ce qui est plausible.
7. Sur donnees simulees, verifier `parent_accuracy_triggered` et la profondeur des
   generations estimees.
8. Lancer plusieurs chaines avec des `rng_seed` differents pour verifier qu'elles
   donnent les memes distributions posterior.

Valeurs indicatives, non garanties :

- MALA : acceptance souvent proche de `0.57` ;
- blocs MH multidimensionnels : environ `0.20` a `0.35` ;
- MH unidimensionnel de `beta` : souvent autour de `0.40`.

## Erreurs frequentes

### `Events outside every spatial domain`

Au moins un evenement est hors de l'union des polygones. Verifier les bornes,
le systeme de coordonnees et `domains.locate(catalog.x, catalog.y)`.

### `Events must be sorted by non-decreasing time`

Trier simultanement toutes les colonnes :

```python
order = np.argsort(t)
catalog = EventCatalog(t[order], x[order], y[order], magnitudes[order])
```

### `The marked ETAS model requires event magnitudes`

Le modele contient `alpha` ou `gamma`, mais le catalogue n'a pas de magnitudes.
Fournir les magnitudes ou utiliser un `ETASParameters` non marque.

### `learn_nu=True is not supported with gp_backend='sparse'`

Utiliser `gp_backend="exact"` ou fixer `learn_nu=False`.

### Chaines lentes ou taux d'acceptation extremes

- acceptance MALA trop haute : augmenter progressivement `mala_step` ;
- acceptance MALA trop basse : diminuer `mala_step` ;
- verifier les traces avant de modifier les priors ;
- ne pas conclure a partir du seul taux d'acceptation.

## Exemple complet de reference

Le script suivant constitue le test SPIN-H principal du depot :

```bash
python -m package.first_test_spinh
```

Il montre la simulation Hawkes, l'inference sparse, les resumes posterior,
l'evaluation de l'intensite conditionnelle et le declustering avec verite connue.
