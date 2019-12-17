# Telegraph Gilliespie algorithm
#
# 	G--k0--> G + M, M --k1--> 0
#    kon||koff
#	G*
#
# 1. Set the initial state N. Initialize time t to zero.
# 2. Calculate the reaction propensities ak(N).
# 3. Draw a sample Rk from the random variable R (Figure 7.43).
# 4. Draw a sample τ from the random variable T (Figure 7.44).
# 5. Increment the simulation time t → t + τ to account for the elapsed time.
# 6. Update the state vector N → N + sk to reflect the fact that reaction Rk has occurred.
# 7. Return to step 2.

import random
import numpy as np
from matplotlib import pyplot as plt


def get_samples(n_samples, initial_M, num_reactions, k0, k1, kon, koff, initial_G_on, time=None, seed=None):
	'''Generates simulated paths for the Telegraph model using Gillespie's algorithm.

	parameters:
		n_samples - number of sample paths that will be generated.
		initial_M - the initial number of mRNA copies that each path will start with
		num_reactions - the number of reactions that will be simulated per sample path
		k0 - the rate parameter for mRNA generation in the Telegraph model
		k1 - the rate paraneter for mRNA decay in the Telegraph model
		kon - the rate paraneter for the gene turning on in the Telegraph model
		koff - the rate paraneter for the gene turning off in the  Telegraph model
		initial_G_on - the proportion of the samples which will start with the gene on
	returns:
		si_tMG - a list containing 'n_samples' sample paths. Each sample path is itself a list containing
		'num_reactions' tuples. Each tuple is lenght 3, with its first element being the time at which
		a reaction occured, its second element being the number of mRNA now present after this
		reaction, and the third being the state of the gene, True for on, and False for off.
	'''
	# if seed parameter is passed to function, set seed.
	if seed!=None:
		random.seed(seed)

	a0 = k0
	a1 = lambda M: M*k1
	aon = kon
	aoff = koff

	si_tMG = []
	for i in range(n_samples):
		r = random.uniform(0,1)
		if r<initial_G_on:
			si_tMG.append([(0,initial_M,True)])
		else:
			si_tMG.append([(0,initial_M,False)])
		

	for s in range(n_samples):
		i=0
		while True:
			# If G is active
			if si_tMG[s][i][2]:

				# Which interaction happened?
				A = [a0, a1(si_tMG[s][-1][1]),aoff]
				r1 = random.uniform(0,sum(A))
				cumsum_a=0
				for k,a in enumerate(A):
					cumsum_a+=a
					if r1 < cumsum_a:
						interaction = k
						break

				# Interarrival time
				r2 = random.uniform(0,1)
				T = -np.log(1-r2)/sum(A)
				t = si_tMG[s][-1][0]+T

				# If M was made
				if interaction == 0:
					si_tMG[s]=si_tMG[s]+[(t,si_tMG[s][-1][1]+1,True)]
				# if M decayed
				elif interaction == 1:
					si_tMG[s]=si_tMG[s]+[(t,si_tMG[s][-1][1]-1,True)]
				# if G turned off
				else:
					si_tMG[s]=si_tMG[s]+[(t,si_tMG[s][-1][1],False)]

			# if G is not active
			else:

				# Which interaction happened?
				A = [a1(si_tMG[s][-1][1]),aon]
				r1 = random.uniform(0,sum(A))
				interaction = int(r1>A[0])

				# Interarrival time
				r2 = random.uniform(0,1)
				T = -np.log(1-r2)/sum(A)
				t = si_tMG[s][-1][0]+T

				# Which interaction happened?

				# If M decayed
				if interaction == 0:
					si_tMG[s]=si_tMG[s]+[(t,si_tMG[s][-1][1]-1,False)]
				# If G turned on
				else:
					si_tMG[s]=si_tMG[s]+[(t,si_tMG[s][-1][1],True)]
			i+=1
			if time!=None:
				if t>=time:
					break
			elif i>=num_reactions:
				break
	return si_tMG

def plot(n_samples, initial_M, num_reactions, k0, k1, kon, koff, initial_G_on, plt_samples=False, plt_analytical=True, plt_mean_variance=True, save=False, time=None, title=None, y_lim=None, seed=None, fontsize=12, axes=True):
	si_tMG = get_samples(n_samples, initial_M, num_reactions, k0, k1, kon, koff, initial_G_on, time=time, seed=seed)
	# Find the latest time to which all sample paths have reached
	if time!=None:
		t_final_min=time
	else:
		t_final = [tn[-1][0] for tn in si_tMG]
		t_final_min = min(t_final)

	# Create range of times to calculate mean and variance at
	T = np.arange(0,t_final_min,0.01)

	if plt_samples:
		for j in range(n_samples):
			plt.step([tMG[0] for tMG in si_tMG[j]], [tMG[1] for tMG in si_tMG[j]], where='post')

	if plt_mean_variance:
		# Calculate sample means and variances
		means = []
		variances = []
		for t in T:
			mean = 0
			variance = 0
			for s in range(n_samples):
				for i in range(num_reactions):
					if si_tMG[s][i][0]>t:
						break
				mean += si_tMG[s][i-1][1]
				variance += (si_tMG[s][i-1][1])**2
			mean /= n_samples
			variance -= n_samples*(mean**2)
			variance /= (n_samples-1)
			means.append(mean)
			variances.append(variance)

		# Plot sample mean and variance
		plt.plot(T,means, linewidth=2.0, color='b', label='sample mean')
		plt.plot(T,variances, linewidth=2.0, color='k', label='sample variance')

	if plt_analytical:
		# Analytical solutions
		# Define constants
		A = k0/k1*kon/(kon+koff)
		B = (k0*initial_G_on-k1*A)/(k1-(kon+koff))
		C = initial_M-A-B

		# Define analytical mean
		analytic_mean = lambda t: A+B*np.exp(-(k0+koff)*t)+C*np.exp(-k1*t)

		# Plot analytical mean
		plt.plot(T,[analytic_mean(t) for t in T], linewidth=2.0, color='r', label='analytic mean')

		# Define and plot long time limit of variance
		var=(k0*kon*(k0*koff+(koff+kon)*(k1+koff+kon)))/(k1*(koff+kon)**2*(k1+koff+kon))
		plt.plot(T,[var for t in T], linewidth=2.0, color='g', label='steady state analytic variance')

	# Add limit the time axis to those times for which all sample paths have been calculated
	plt.xlim(0, t_final_min)
	if y_lim!=None:
		plt.ylim(0, y_lim)
	if title!=None:
		plt.title(title, fontsize=fontsize)
	# Add x,y labels
	if axes:
		plt.xlabel('Time')
		plt.ylabel('Number of mRNA')
	# Depending on save flag - save the plot in current working directory
	if save:
		if title!=None:
			plt.savefig(str(title) + '-T-SSA.svg')
		else:	
			plt.savefig('T-SSA.svg')
	# Show plot
	plt.show()
	return plt.gcf(), plt.gca()

def fpt(n_samples, num_reactions, k0, k1, kon, koff, initial_G_on, save=False):
	'''Generates and plots a sample histogram for the first passage time to mRNA production.
	
	parameters:
		n_samples - the number of samples we simulate.
		num_reactions - the number of reactions we run each simulation for
		k0 - the rate parameter for mRNA generation in the Telegraph model
		k1 - the rate paraneter for mRNA decay in the Telegraph model
		kon - the rate paraneter for the gene turning on in the Telegraph model
		koff - the rate paraneter for the gene turning off in the  Telegraph model
		initial_G_on - the proportion of samples we begin with the gene initially on
	Samples begin with the gene on with probability initial_G_on.
	returns: None
	'''
	# Get samples
	si_tMG = get_samples(n_samples, 0, num_reactions, k0, k1, kon, koff, initial_G_on)

	# Finding first passage time 0->1 histogram
	fp_time = []
	for s in range(n_samples):
		for i in range(num_reactions):
			if si_tMG[s][i][1]>0:
				fp_time.append(si_tMG[s][i][0])
				break
	plt.hist(fp_time,bins=50, density=True, label='sample histogram')

	# Analytic first passage time 0->1 distribution
	# Define constants
	A = kon + koff + k0
	B = np.sqrt(-4*kon*k0+A**2)
	C = kon+koff-k0

	# Define density function
	fp_dens = lambda t: (k0*kon)/(2*(kon+koff)*B)*((B-C)*np.exp(-0.5*(A+B)*t)+(B+C)*np.exp(-0.5*(A-B)*t))

	# Find maximum fpt
	t_final = max(fp_time)

	# Plot density function
	T = np.arange(0,t_final,0.01)
	plt.plot(T,[fp_dens(t) for t in T], linewidth=2.0, color='r', label='analytic density')
	plt.legend()

	# Depending on save flag - save the plot in current working directory
	if save:
		plt.savefig('T-fpt-SSA.svg')

	plt.show()

# Define parameters
n_samples = 1
initial_M = 0
num_reactions = 1000
k0 = 100
k1 = 1
kon = 2
koff = 100
initial_G_on = 1

# UNCOMMENT TO GET figure 7 from report
#plot(n_samples, initial_M, num_reactions, k0, k1, kon, koff, initial_G_on, plt_samples=True, plt_mean_variance=False, plt_analytical=False, save=True, seed=1000)

#fpt(n_samples, num_reactions, k0, k1, kon, koff, initial_G_on)
