# Birth death Gilliespie algorithm
#
# --k0--> X
# X --k1-->
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

def get_samples(n_samples, initial_N, num_reactions, k0, k1):
	'''Generates simulated paths for the Birth-Death model using Gillespie's algorithm.

	parameters:
		n_samples - number of sample paths that will be generated.
		initial_N - the initial number of mRNA copies that each path will start with
		num_reactions - the number of reactions that will be simulated per sample path
		k0 - the rate parameter for mRNA generation in the Birth-Death model
		k1 - the rate paraneter fir mRNA decay in the Birth-Death model
	returns:
		jtN - a list containing 'n_samples' sample paths. Each sample path is itself a list containing
		'num_reactions' tuples. Each tuple is a pair, with its first element being the time at which
		a reaction occured, and its second element being the number of mRNA now present after this
		reaction.
	'''
	# Create variables/functions for calculating probability of next reaction type and time waited.
	a0 = k0
	a1 = lambda N: N*k1

	# Create the container for samples with initial conditions set for each sample.
	jtN = [[(0,initial_N)]]*n_samples

	# Run sample path simulation for n_samples
	for j in range(n_samples):

		# Get next reaction and time it occured 'num_reactions' times
		for i in range(num_reactions):

			# Get ratios for calculating reaction and time probabilities
			A = [a0, a1(jtN[j][i][1])]
			a = sum(A)

			# Get 2 independent random numbers
			r1 = random.uniform(0,1)
			r2 = random.uniform(0,1)

			# Find the change in the number of mRNA due to which reaction occured using r1 and A
			R = int(r1 < A[0]/a)*2 -1

			# Find the time elapsed from the previous reaction to this reaction using r2 and A
			T = -np.log(1-r2)/a

			# Generate tuple for the new time and new number of mRNA. Add these to list.
			t = jtN[j][-1][0]+T
			N = jtN[j][-1][1]+R
			jtN[j] = jtN[j]+[(t,N)]

	# Return the list of sample paths
	return jtN

def plot(n_samples, initial_N, num_reactions, k0, k1, plt_samples=True, plt_mean=False, plt_variance=False, save=False):
	# Get samples
	jtN = get_samples(n_samples, initial_N, num_reactions, k0, k1)

	# if the plt_samples flag is True, then plot every sample path
	if plt_samples:
		for j in range(n_samples):
			plt.step([tn[0] for tn in jtN[j]], [tn[1] for tn in jtN[j]], where='post')

	# Calculate the latest time that all sample paths reached
	t_final = [tn[-1][0] for tn in jtN]
	t_final_min = min(t_final)

	# If we are plotting either the mean or variance plotting flags are True...
	if plt_mean or plt_variance:

		# Set the time values at which we wish to calculate the sample mean and variance
		x = np.arange(0,t_final_min,0.001)

		# Create containers for mean and variances
		means = []
		variances = []
		
		# For each time
		for k in x:
			# Set variables to keep track of sum
			mean = 0
			variance = 0
			# For each sample path... 
			for j in range(n_samples):
				# ...find the last reaction that occured prior to the time
				for i in range(num_reactions):
					if jtN[j][i][0]>k:
						break
				# ...add the number of mRNA at this last reaction to our mean and variance
				# since this will be the the same as the number of mRNA at the time we are interested in.
				mean += jtN[j][i-1][1]
				variance += (jtN[j][i-1][1])**2
			# Final calculations to find sample mean/variance
			mean /= n_samples
			variance -= (mean**2)*n_samples
			variance /= (n_samples-1)
			# Append these to the list of means and variances
			means.append(mean)
			variances.append(variance)

		# If flags for plotting mean/variance are set, then we plot them.
		if plt_mean:
			plt.plot(x,means, linewidth=2.0, color='#fffcb3', label='mean')
		if plt_variance:
			plt.plot(x,variances, linewidth=2.0, color='k', label='variance')
	# Limit the time axis to those times for which all sample paths have been calculated
	plt.xlim(0, t_final_min)
	# Axis labels
	plt.xlabel('Time')
	plt.ylabel('Number of mRNA')
	# Depending on save flag - save the plot in current working directory
	if save:
		plt.savefig('BD-SSA.svg')
	# Show plot
	plt.show()

# Set parameters and plot sample paths, mean and variance.
n_samples = 1
initial_N = 0
num_reactions = 1500
k0 = 20
k1 = 1

#UNCOMMENT THIS TO GET FIGURE 4 from report
random.seed(1234)
plot(1, 0, 1500, 20, k1, plt_samples=True, plt_mean=False, plt_variance=False, save=True)
