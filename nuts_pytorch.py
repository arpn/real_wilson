"""
This package implements the No-U-Turn Sampler (NUTS) algorithm 6 from the NUTS
paper (Hoffman & Gelman, 2011).

Content
-------

The package mainly contains:
  nuts6                     return samples using the NUTS
  test_nuts6                example usage of this package

and subroutines of nuts6:
  build_tree                the main recursion in NUTS
  find_reasonable_epsilon   Heuristic for choosing an initial value of epsilon
  leapfrog                  Perfom a leapfrog jump in the Hamiltonian space
  stop_criterion            Compute the stop condition in the main loop


A few words about NUTS
----------------------

Hamiltonian Monte Carlo or Hybrid Monte Carlo (HMC) is a Markov chain Monte
Carlo (MCMC) algorithm that avoids the random walk behavior and sensitivity to
correlated parameters, biggest weakness of many MCMC methods. Instead, it takes
a series of steps informed by first-order gradient information.

This feature allows it to converge much more quickly to high-dimensional target
distributions compared to simpler methods such as Metropolis, Gibbs sampling
(and derivatives).

However, HMC's performance is highly sensitive to two user-specified
parameters: a step size, and a desired number of steps.  In particular, if the
number of steps is too small then the algorithm will just exhibit random walk
behavior, whereas if it is too large it will waste computations.

Hoffman & Gelman introduced NUTS or the No-U-Turn Sampler, an extension to HMC
that eliminates the need to set a number of steps.  NUTS uses a recursive
algorithm to find likely candidate points that automatically stops when it
starts to double back and retrace its steps.  Empirically, NUTS perform at
least as effciently as and sometimes more effciently than a well tuned standard
HMC method, without requiring user intervention or costly tuning runs.

Moreover, Hoffman & Gelman derived a method for adapting the step size
parameter on the fly based on primal-dual averaging.  NUTS can thus be used
with no hand-tuning at all.

In practice, the implementation still requires a number of steps, a burning
period and a stepsize. However, the stepsize will be optimized during the
burning period, and the final values of all the user-defined values will be
revised by the algorithm.

reference: arXiv:1111.4246
"The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
Carlo", Matthew D. Hoffman & Andrew Gelman
"""

"""
The MIT License (MIT)

Copyright (c) 2012 Morgan Fouesneau

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Changed to use PyTorch by Antti Honkela, 2018-10-10
Updated copy-construction for newer versions of Pytorch
and added a progress bar by Arttu Pönni, 2022-08-12
"""
import torch
import math
import random
from tqdm import tqdm


def leapfrog(theta, r, grad, epsilon, f):
    """ Perfom a leapfrog jump in the Hamiltonian space
    INPUTS
    ------
    theta: tensor[double, ndim=1]
        initial parameter position

    r: tensor[double, ndim=1]
        initial momentum

    grad: double
        initial gradient value

    epsilon: double
        step size

    f: callable
        it should return the log probability evaluated at theta
        logp = f(theta)

    OUTPUTS
    -------
    thetaprime: tensor[double, ndim=1]
        new parameter position
    rprime: tensor[double, ndim=1]
        new momentum
    gradprime: double
        new gradient
    """
    device = theta.device
    dtype = torch.double
    # make half step in r
    rprime = r + 0.5 * epsilon * grad
    thetaprime0 = theta.data + epsilon * rprime.data
    # make new step in theta
    thetaprime = thetaprime0.clone().detach().to(dtype=dtype, device=device).requires_grad_(True)
    #compute new gradient
    logpprime = f(thetaprime)
    logpprime.backward()
    gradprime = thetaprime.grad
    # make half step in r again
    rprime = rprime + 0.5 * epsilon * gradprime
    return thetaprime.detach(), rprime, gradprime, logpprime


def find_reasonable_epsilon(theta0, grad0, logp0, f):
    """ Heuristic for choosing an initial value of epsilon """
    device = theta0.device
    dtype = torch.double
    epsilon = 1.
    r0 = torch.normal(torch.zeros(len(theta0), device=device, dtype=dtype), 1.)

    # Figure out what direction we should be moving epsilon.
    _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, epsilon, f)
    # brutal! This trick make sure the step is not huge leading to infinite
    # values of the likelihood. This could also help to make sure theta stays
    # within the prior domain (if any)
    k = 1.
    while torch.isinf(logpprime) or torch.isinf(gradprime).any():
        k *= 0.5
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon * k, f)

    epsilon = 0.5 * k * epsilon

    acceptprob = torch.exp(logpprime - logp0 - 0.5 * (rprime.dot(rprime) - r0.dot(r0)))

    a = 2. * float((acceptprob > 0.5)) - 1.
    # Keep moving epsilon in that direction until acceptprob crosses 0.5.
    while ( (acceptprob ** a) > (2. ** (-a))):
        epsilon = epsilon * (2. ** a)
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon, f)
        acceptprob = torch.exp(logpprime - logp0 - 0.5 * (rprime.dot(rprime) - r0.dot(r0)))

    print("find_reasonable_epsilon =", epsilon)

    return epsilon


def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    """ Compute the stop condition in the main loop
    dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

    INPUTS
    ------
    thetaminus, thetaplus: tensor[double, ndim=1]
        under and above position
    rminus, rplus: tensor[double, ndim=1]
        under and above momentum

    OUTPUTS
    -------
    criterion: bool
        return if the condition is valid
    """
    dtheta = thetaplus - thetaminus
    return (dtheta.dot(rminus) >= 0) & (dtheta.dot(rplus) >= 0)


def build_tree(theta, r, grad, logu, v, j, epsilon, f, joint0):
    """The main recursion."""
    if (j == 0):
        # Base case: Take a single leapfrog step in the direction v.
        thetaprime, rprime, gradprime, logpprime = leapfrog(theta, r, grad, v * epsilon, f)
        joint = logpprime - 0.5 * rprime.dot(rprime)
        # Is the new point in the slice?
        nprime = int(logu < joint)
        # Is the simulation wildly inaccurate?
        sprime = int((logu - 1000.) < joint)
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        gradminus = gradprime[:]
        gradplus = gradprime[:]
        # Compute the acceptance probability.
        alphaprime = min(1., torch.exp(joint - joint0).item())
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime = build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaminus, rminus, gradminus, logu, v, j - 1, epsilon, f, joint0)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaplus, rplus, gradplus, logu, v, j - 1, epsilon, f, joint0)
            # Choose which subtree to propagate a sample up from.
            if (torch.rand(1) < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                thetaprime = thetaprime2[:]
                gradprime = gradprime2[:]
                logpprime = logpprime2
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime


def nuts6(f, M, Madapt, theta0, delta=0.6):
    """
    Implements the No-U-Turn Sampler (NUTS) algorithm 6 from from the NUTS
    paper (Hoffman & Gelman, 2011).

    Runs Madapt steps of burn-in, during which it adapts the step size
    parameter epsilon, then starts generating samples to return.

    Note the initial step size is tricky and not exactly the one from the
    initial paper.  In fact the initial step size could be given by the user in
    order to avoid potential problems

    INPUTS
    ------
    f: callable
        it should return the log probability evaluated at theta
        logp = f(theta)

    M: int
        number of samples to generate.

    Madapt: int
        the number of steps of burn-in/how long to run the dual averaging
        algorithm to fit the step size epsilon.

    theta0: tensor[double, ndim=1]
        initial guess of the parameters.

    KEYWORDS
    --------
    delta: float
        targeted acceptance fraction

    OUTPUTS
    -------
    samples: tensor[double, ndim=2]
    M x D matrix of samples generated by NUTS.
    note: samples[0, :] = theta0
    """
    
    if len(theta0.shape) > 1:
        raise ValueError('theta0 is expected to be a 1-D array')

    dtype = torch.double
    device = theta0.device
    D = len(theta0)
    samples = torch.empty((M + Madapt, D), dtype=dtype, device=device)
    lnprob = torch.empty(M + Madapt, dtype=dtype, device=device)

    theta = theta0.clone().detach().to(dtype=dtype, device=device).requires_grad_(True)
    logp = f(theta)
    logp.backward()
    grad = theta.grad
    samples[0, :] = theta0
    lnprob[0] = logp

    # Choose a reasonable first epsilon by a simple heuristic.
    epsilon = find_reasonable_epsilon(theta, grad, logp, f)

    # Parameters to the dual averaging algorithm.
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = math.log(10. * epsilon)

    # Initialize dual averaging algorithm.
    epsilonbar = torch.ones(1, device=device, dtype=dtype)
    Hbar = 0

    for m in tqdm(range(1, M + Madapt)):
        # Resample momenta.
        r0 = torch.normal(torch.zeros(D, device=device, dtype=dtype), 1.0)

        #joint lnp of theta and momentum r
        joint = logp - 0.5 * r0.dot(r0)

        # Resample u ~ uniform([0, exp(joint)]).
        # Equivalent to (log(u) - joint) ~ exponential(1).
        logu = joint - torch.distributions.exponential.Exponential(
            torch.ones(1, device=device, dtype=dtype)).sample()

        # if all fails, the next sample will be the previous one
        samples[m, :] = samples[m - 1, :]
        lnprob[m] = lnprob[m - 1]

        # initialize the tree
        thetaminus = samples[m - 1, :]
        thetaplus = samples[m - 1, :]
        rminus = r0[:]
        rplus = r0[:]
        gradminus = grad[:]
        gradplus = grad[:]

        j = 0  # initial heigth j = 0
        n = 1  # Initially the only valid point is the initial point.
        s = 1  # Main loop: will keep going until s == 0.

        while (s == 1):
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (random.random() < 0.5) - 1)

            # Double the size of the tree.
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint)

            # Use Metropolis-Hastings to decide whether or not to move to a
            # point from the half-tree we just generated.
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (torch.rand(1, device=device, dtype=dtype) < _tmp):
                samples[m, :] = thetaprime[:]
                lnprob[m] = logpprime
                logp = logpprime
                grad = gradprime[:]
            # Update number of valid points we've seen.
            n += nprime
            # Decide if it's time to stop.
            s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            # Increment depth.
            j += 1

        # Do adaptation of epsilon if we're still doing burn-in.
        eta = 1. / float(m + t0)
        Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
        if (m <= Madapt):
            epsilon = math.exp(mu - math.sqrt(m) / gamma * Hbar)
            eta = m ** -kappa
            epsilonbar = math.exp((1. - eta) * math.log(epsilonbar) + eta * math.log(epsilon))
        else:
            epsilon = epsilonbar
    samples = samples[Madapt:, :]
    lnprob = lnprob[Madapt:]
    print('Final epsilon =', epsilon)
    return samples    # , lnprob, epsilon


def test_nuts6():
    """ Example usage of nuts6: sampling a 2d highly correlated Gaussian distribution """
    device = 'cpu'
    dtype = torch.double
    
    def correlated_normal(theta):
        """
        Example of a target distribution that could be sampled from using NUTS.
        (Although of course you could sample from it more efficiently)
        Doesn't include the normalizing constant.
        """

        # Precision matrix with covariance [1, 1.98; 1.98, 4].
        # A = np.linalg.inv( cov )
        A = torch.tensor([[50.251256, -24.874372],
                          [-24.874372, 12.562814]],
                         dtype=dtype, device=device)

        grad = -theta @ A
        logp = 0.5 * torch.dot(grad, theta.T)
        return logp

    D = 2
    M = 5000
    Madapt = 5000
    theta0 = torch.normal(torch.zeros(D, device=device, dtype=dtype), 1.0)
    delta = 0.2

    mean = torch.zeros(2, device=device, dtype=dtype)
    cov = torch.tensor([[1, 1.98],
                        [1.98, 4]], device=device, dtype=dtype)

    print('Running HMC with dual averaging and trajectory length %0.2f...' % delta)
    samples = nuts6(correlated_normal, M, Madapt, theta0, delta)
    #print('Done. Final epsilon = %f.' % epsilon)

    samples = samples[1::10, :]
    print('Percentiles')
    print(torch.percentile(samples, [16, 50, 84], axis=0))
    print('Mean')
    print(torch.mean(samples, axis=0))
    print('Stddev')
    print(torch.std(samples, axis=0))

    import pylab as plt
    import numpy.random as npr
    temp = npr.multivariate_normal(mean.numpy(), cov.numpy(), size=500)
    plt.plot(temp[:, 0], temp[:, 1], '.')
    plt.plot(samples[:, 0].numpy(), samples[:, 1].numpy(), 'r+')
    plt.show()
