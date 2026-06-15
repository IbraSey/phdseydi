
# BUG REPORT : NUMERICAL OVERFLOW OF LATENT POISSON PROCESS SIZE 

Appplying the Gibbs.run() method of the SSGC_Gibbs class to a real-life use case, results in a numerical overflow of the size of the latent homogeneous Poisson process (PP). Recall that the size is a random variable, which follows a Poisson distribution whose intensity is modeled a priori by a Gamma distribution, to benefit from conditional conjugacy.

There are at least three reasons expaining this overflow:

1. Recall that the mean size of the latent PP is taken equal to an upper bound of the zones effects, which in turn is an upper bound of the inhomogenous intensity of the observed PP, multiplied by the the volume of the time-space search domain, the latter volume being of the order of 10^7. Oversizing the intensity's upper bound can have critical consequences, since the size of the MCMC chain varies from one iteration to another, in direct relation to the size of the latent variable block. A possible amelioration could be to re-define the latent PP as being piecewise homogeneous on the zones of the area source model.

2. The above problem can be amplified by the the lack of identifiability of the spatially informed Sigmoid GP intensity. More precisely, one can multiply the zones' effects by an arbitrary factor > 1, and re-define the latent GP trajectory to obtain an identical intensity function (exercise). Thus, in practice, nothing prevents the zone's effects to diverge during the MCMC iterations, leading to the said overflow.

3. Another possible reason, could be that the Gamma prior on the latent Poisson process size is too vague. Because the MCMC is initialized using prior draws, this could lead to unrealistically large draws of the Poisson mean parameter. So another possible response would be to used better-contrained priors, even possibly imposing a prior upper bound on the latent PP size.









