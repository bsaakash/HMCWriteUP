Hamiltonian Monte Carlo (HMC) algorithm
=======================================
The description of the algorithm here mainly follows the presentation in [Neal2011]_. 


Hamiltonian dynamics
--------------------
The state of a dynamic system in described by its position :math:`p`, and its momentum :math:`q`. When Hamiltonian dynamics is used in MCMC, then the position corresponds to the variables of interest, the potential energy of the system is the negative of the log-likelihood of the probability density function of these variables, and a momentum variable has to be artificially introduced for each of the position variables.

Hamilton's equations
++++++++++++++++++++
Let the dimension of the position vector :math:`q` be :math:`d`. Then, the momentum :math:`p` is another :math:`d` dimensional vector. Hence, the state space of the system has :math:`2d` dimensions. The state of the system is described by a function of the position and momentum variables known as the **Hamiltonian**, :math:`H(q, p)`.

According to Hamilton's equations, the partial derivatives of the Hamiltonian determine how the position and momentum variables change over time, :math:`t`:

.. math::
   \frac{\partial q_i}{\partial t} &= \frac{\partial H}{\partial p_i} \\
   \frac{\partial p_i}{\partial t} &= -\frac{\partial H}{\partial q_i}

for :math:`i = 1,\ldots, d`. These equations describe the motion of the system.

Potential and kinetic energy
++++++++++++++++++++++++++++
For HMC, Hamiltonian functions that can be written as the sum of the potential energy :math:`U(q)` and kinetic energy :math:`K(p)` are used. 

.. math::
   H(q, p) = U(q) + K(p)
   

The potential energy :math:`U(q)` is defined to be the negative of the log probability density of the distribution for :math:`q` that we wish to draw samples from. 

The kinetic energy is usually defined as

.. math::
   K(p) = \frac{p^T M^{-1} p}{2}
   
Here, M is a symmetric, positive definite matrix known as the mass matrix, which is typically a diagonal matrix, and often a scalar multiple of the identity matrix. This form of the kinetic energy corresponds to the negative of the log probability density of a zero-mean Gaussian distribution with a covariance matrix :math:`M`.


Therefore, Hamilton's equations can be written as:

.. math::
   \frac{\partial q_i}{\partial t} &= [M^{-1}p]_i \\
   \frac{\partial p_i}{\partial t} &= -\frac{\partial U}{\partial q_i}
   
If the kinetic energy is assumed to be of the form :math:`K(p) = \frac{p^T M^{-1} p}{2}`, where :math:`M` is diagonal, then, 

.. math::
   K(p) = \sum_{i=1}^d \frac{p^2_i}{2m_i}

Discretizing Hamilton's equations
+++++++++++++++++++++++++++++++++
For computer implementation, Hamilton's equations must be approximated by discretizing time using a small step size :math:`\epsilon`. The state of the system is computed by integrating Hamilton's equations using the **Leap frog** method, which is given by:

.. math::
   p_i(t + \epsilon/2) &= p_i(t) - \frac{\epsilon}{2} \frac{\partial U(q(t))}{\partial q_i} \\
   q_i(t + \epsilon) &= q_i(t) + \epsilon \frac{p_i(t + \epsilon/2)}{m_i} \\
   p_i(t + \epsilon) &= p_i(t + \epsilon/2) - \frac{\epsilon}{2} \frac{\partial U(q(t + \epsilon))}{\partial q_i} 



MCMC from Hamiltonian dynamics
------------------------------
To use Hamiltonian dynamics for MCMC sampling, the target probability density function has to be translated into a potential energy function (which is the negative log of the target probability density function), and a kinetic energy function has to be introduced.


Canonical distribution
++++++++++++++++++++++
Borrowing the terminology from statistical mechanics, given an energy function :math:`E(x)`, where the state of the system is represented as :math:`x`, the probability density function of the *canonical distribution* over the states is given by 

.. math::
   P(x) = \frac{1}{Z} \exp{\frac{-E(x)}{T}}
   
where, T is the temperature of the system, and Z is a normalizing constant. 

Conversely, a distribution of interest :math:`P(x)`, can be obtained as a canonical distribution with :math:`T=1` by setting :math:`E(x) = -\log{P(x)} - \log{Z}`, where, :math:`Z` is any convenient positive constant.

The Hamiltonian :math:`H(q, p)` is an energy function for the joint state of position and momentum and so, this energy function defines a joint distribution for these variables 

.. math::
   P(q, p) = \frac{1}{Z} \exp{\frac{-H(q, p)}{T}}.
   
If :math:`H(q, p) = U(q) + K(p)`, then, 

.. math::
   P(q, p) = \frac{1}{Z} \exp{\frac{-U(q)}{T}} \exp{\frac{-K(p)}{T}}
   :label: canonicalDist


This shows that :math:`q` and :math:`p` are indepenedent and have their own canonical distributions with energy functions :math:`U(q)` and :math:`K(p)` respectively.

For MCMC, we can express the posterior probability distibution as a canonical distribution with :math:`T=1`, using a potential energy function that is given by:

.. math::
   U(q) = -\log{\pi(q)L(q|D)}
   :label: potentialEnergy

where, :math:`\pi(q)` is the prior probability denstity and :math:`L(q|D)` is the likelihood function.

Hamiltonian Monte Carlo algorithm
+++++++++++++++++++++++++++++++++
Conditions for using HMC:

- Only continuous distributions for which the density function can be evaluated can be sampled from
- The partial derivatives of the log of the density function must be computable

HMC draws samples from the canonical distribution for :math:`q` and :math:`p` given in Equation :eq:`canonicalDist`. The potential energy is as given in Equation :eq:`potentialEnergy`. We can choose the distribution of the momentum variables, which are independent of the position variables, and specify the distribution via the kinetic energy function :math:`K(p)`. The current practice is to use a quadratic kinetic energy with HMC, which leads to a zero-mean multivariate Gaussian distribution for the momentum variables. Typically, the components of the momentum vector are specified to be independent, with component :math:`i` having a variance :math:`m_i`. Then, the kinetic energy function producing this distribution is given by:

.. math::
   K(p) = \sum_{i=1}^d \frac{p_i^2}{2m_i}
   
There are two steps in the HMC algorithm:

- In the fist step, new values for the momentum are drawn at random from their Gaussian distribution, independently of the current values of the position variables. 

- In the second step, a Metropolis update is performed, using Hamiltonian dynamics to propose a new state. Starting from the current state, Hamiltonian dynamics is simulated for :math:`L` steps using the leap frog algorithm with step size :math:`\epsilon`. :math:`L` and :math:`\epsilon` are parameters of the algorithm which must be tuned for good performance. Then, the momentum variables are negated at the end of the :math:`L` steps, to obtain a proposed state :math:`(q^*, p^*)`. This proposed state is accepted as the next state of the Markov chain with a probability that is given by

.. math::
   \min{[1, \exp{(-H(q^*, p^*) + H(q, p))}]} = \min{[1, \exp{(-U(q^*) + U(q) - K(p^*) + K(p))}]}
   
If the proposed state is not acepted, the next state is the same as the current state. 

The algorithm is given below:

.. code-block:: 

	HMC = function (U, grad_U, epsilon, L, current_q)
	{
	 q = current_q
	 p = rnorm(length(q),0,1)  # independent standard normal variates
	 current_p = p
		
	# Make a half step for momentum at the beginning 
	p = p - epsilon * grad_U(q) / 2
	
	# Alternate full steps for position and momentum
	 for (i in 1:L)
	 {
	
	   # Make a full step for the position
	   q = q + epsilon * p
	   # Make a full step for the momentum, except at end of trajectory 
	   if (i!=L) p = p - epsilon * grad_U(q)
	}
	# Make a half step for momentum at the end.
	p = p - epsilon * grad_U(q) / 2
	# Negate momentum at end of trajectory to make the proposal symmetric 
	p = -p
	
	# Evaluate potential and kinetic energies at start and end of trajectory
	current_U = U(current_q) 
	current_K = sum(current_pˆ2) / 2 
	proposed_U = U(q)
	proposed_K = sum(pˆ2) / 2
	
	# Accept or reject the state at end of trajectory, returning either
	# the position at the end of the trajectory or the initial position
	if (runif(1) < exp(current_U-proposed_U+current_K-proposed_K)) {
	      return (q)  # accept
	    }
	else {
	      return (current_q)  # reject
	    }
	}

As seen in the code block displayed above, the HMC function takes two functions as its first two arguments. The first argument ``U`` is a function that returns the potential energy given an input :math:`q`, and the second argument ``grad_U`` is a function that returns the vector of partial derivaties of :math:`U` given an input :math:`q`. The next two arguments ``epsilon`` and ``L`` define the step size and the number of steps of the leap frog algorithm for simulating the Hamiltonian dynamics, and the last argument ``current_q`` defines the current position from where the trajectory starts. 


Computing the gradient of the potential energy
----------------------------------------------
The potential energy :math:`U(q)` is given by Equation :eq:`potentialEnergy`. It is the negative of the log of the posterior probability density function, i.e., :math:`U(q) = -\log{\pi(q)L(q|D)}`. In order to compute the gradient of the potential energy, we need

.. math::
   \frac{\partial U(q)}{\partial q} &= - \frac{\partial(\log{(\pi(q)L(q|D)))}}{\partial q} \\
   &=- \frac{\partial (\log{\pi(q) + \log{L(q|D)}})}{\partial q} \\
   &= - \frac{\partial \log{\pi(q)}}{\partial q} - \frac{\partial \log{L(q|D)}}{\partial q} \\
   &= - \frac{\partial \log{\pi(q)}}{\partial q} - \frac{\partial f(g(q), D, \Sigma)}{\partial q} \\
   &= - \frac{\partial \log{\pi(q)}}{\partial q} - \frac{\partial f(g(q), D, \Sigma)}{\partial g} \frac{\partial g(q)}{\partial q}\\

In the equations above, the log likelihood has been represented as a function :math:`f(g(q), D, \Sigma)` that computes the likelihood given the prediction from the model :math:`g(q)`, the data :math:`D`, and the covariance of the error terms :math:`\Sigma`. 

As seen in the last of the equations above, the gradient :math:`\frac{\partial g(q)}{\partial q}` of the model predictions :math:`g(q)` with respect to the parameters :math:`q` is required to be able to compute the gradient of the potential energy, which is required in HMC.


Methods to compute the gradient of the model prediction
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

- Finite difference methods 

	+ works with a black-box model
	+ but is computationally expensive to the extent of being impractical
	+ and can lack accuracy (susceptible to discretization error or round off error)

- **Direct differentiation method**

	+ makes use of the information contained in the system of equations
	+ can provide **accurate** (exact) values of the gradient
	+ is **efficient** does not resort to perturbation like finite difference schemes
	+ need access simulation source code to implement this method, must be fully integrated into the simulation program

- Adjoint variable method

	+ has the same advantages as the direct differentiation method
	+ can be more efficient when the number of design variables :math:`\dim(q)` is greater than the number of performance measures :math:`f(g(q))`
	
- Automatic differentiation method

.. + make use of the information contained in the system of equations,
.. + can provide accurate (exact) values of the gradient,
.. + do not resort to perturbation like finite difference schemes and are efficient,



.. [Neal2011] 
   R. M. Neal, “MCMC using Hamiltonian dynamics”, Chapter 5 in *Handbook of Markov Chain Monte Carlo*, (S. Brooks, A. Gelman, G. L. Jones and X.-L. Meng, eds.) CRC Press, New York, 2011.