import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def get_ks(L_data):
	'''Takes a panda dataframe from the csv file for the Larsson et al. dataset and returns 
	three numpy arrays for the kinetic parameters ksyn, koff, kon'''
	A = L_data
	burst_size=A.iloc[:,1]
	burst=[0]*len(burst_size)

	for i in range(0,len(burst_size)):
		burst[i]=burst_size[i][1:-1] #getting rid of brackets
 
	for i in range(0,len(burst)):
		#creatings a list by splitting up everytime there is a space
		burst[i]=burst[i].split(' ')
		burst[i] = list(filter(None, burst[i]))#getting rid of all the empty strings
		for j in range(0,len(burst[i])):
			burst[i][j]=float(burst[i][j])#converting string to float

	kon=[]
	koff=[]
	ksyn=[]        
	for i in range(0,len(burst)):
		kon.append(burst[i][0])
		koff.append(burst[i][1])
		ksyn.append(burst[i][2])

	kon=np.array(kon)
	koff=np.array(koff)
	ksyn=np.array(ksyn)

	gene_names = list(A['Gene'])

	return kon, koff, ksyn, gene_names

def get_alpha_beta_ff(kon, koff, ksyn):
	'''Takes the kinetic parameters and returns three arrays: burstsize alpha,
	the proportion of time that the gene is off i.e. beta, and the fano factor'''
	alpha=ksyn/koff
	beta=koff/(koff+kon)
	var=ksyn*((ksyn*beta*(1-beta))/(1+koff+kon)+(1-beta))
	mean=ksyn*(1-beta)
	ff = var/mean
	return alpha, beta, ff

def get_ff_rel_error(alpha, ff):
	'''Returns an array for the relative error between the true fano factor,
	and the fano factor in the bursty limit'''
	return (1+(alpha-ff))/ff

def plot_ff_ffstar_relerror(alpha, ff):
	'''Plots a histogram for the relative error in the fano factor'''
	rel_error = get_ff_rel_error(alpha, ff)
	print("Proportion of genes for which rel_error<0.2: ", sum(rel_error<0.2)/len(ff))
	counts, bins, bars = plt.hist(rel_error, bins=100, range=(0,2))
	plt.xlabel(r"Relative error between $FF_T$ and $FF_T^*$", fontsize=13)
	plt.ylabel(r"Frequency", fontsize=13)
	plt.savefig('ff_rel_error.svg')
	plt.show()

def get_TATA_cat(O_data, gene_names):
	'''Splits the data set into indices at which the gene has the TATA string,
	and those that don't. Since some gene names are not given in the Larsson
	et al. dataset, we must leave some out. These are returned as another array
	of indices.'''
	name_list = O_data['Gene'].tolist()
	TATA_list = O_data['has_TATA'].tolist()
	d = dict(zip(name_list,TATA_list))
	tata_list_perm = []
	missing = []
	for i,name in zip(range(len(gene_names)),gene_names):
		try:
			tata_list_perm.append(d[name])
		except:
			missing.append(i)
			print(name, " not found")
	tata = np.array(tata_list_perm)
	true_inds =  np.where(tata==True)[0]
	false_inds = np.array(range(len(tata)))
	false_inds = np.delete(false_inds, true_inds)
	cats = [('TATA True', true_inds.astype(int)), ('TATA False',false_inds.astype(int))]
	return cats, missing

def get_gene_length(O_data, gene_names):
	'''Returns an array of lengths of each gene. Again since some gene names are
	not given in the Larsson et al. dataset, we must leave these out. These are
	returned as another array of indices.'''
	# Getting gene lengths in correct order
	name_list = O_data['Gene'].tolist()
	gene_length_list = O_data['gene_length'].tolist()
	d = dict(zip(name_list,gene_length_list))
	gene_length_list_perm = []
	missing = []
	for i,name in zip(range(len(gene_names)),gene_names):
		try:
			gene_length_list_perm.append(d[name])
		except:
			missing.append(i)
			print(name, " not found")
	gene_length = np.array(gene_length_list_perm)
	return gene_length, missing

def gene_length_against_ff_rel_error(alpha, ff, O_data, gene_names):
	'''Plots the gene length against the relative error in the Fano factor. Also
	prints the Pearson coefficient of correlation between the two.'''
	gene_length, missing = get_gene_length(O_data, gene_names)

	ff_error = list(get_ff_rel_error(alpha, ff))
	for i in missing:
		del(ff_error[i])
	ff_error=np.array(ff_error)
	
	print("Correlation coeff btween gene length and FF rel error: ", np.corrcoef([ff_error,gene_length]))

	plt.scatter(	np.log10(ff_error),
			np.log10(gene_length),
			s=3)
	plt.ylabel(r"Log$_{10}$ gene length", fontsize=13)
	plt.xlabel(r"Log$_{10}$ relative error between $FF_T$ and $FF_T^*$", fontsize=13)
	plt.savefig("gene_length_against_ff_rel_error.svg")
	plt.show()

def TATA_ff_rel_error_correlation(alpha, ff, O_data, gene_names):
	'''Plots two normalised histograms of the relative error in the Fano factor. One
	for those genes that contain the TATA string, and one for those who don't. Also
	prints the point-biserial coefficient of correlation between the two groups.'''
	# Make list of 1s for genes with TATA, 0 o/w
	tata_cats, missing = get_TATA_cat(O_data, gene_names)
	has_tata_inds = tata_cats[0][1]
	no_tata_inds = tata_cats[1][1]
	tata = [0]*(len(has_tata_inds)+len(no_tata_inds))
	for i in range(len(tata)):
		if i in has_tata_inds:
			tata[i]=1

	# Delete genes which we don't have TATA knowledge of from ffstar_rel_error list
	ff_error = list(get_ff_rel_error(alpha, ff))
	for i in missing:
		del(ff_error[i])

	ff_error=np.log10(np.array(ff_error))

	fig, ax = plt.subplots(2,1)
	fig.text(0.04, 0.5, r'Frequency', va='center', rotation='vertical', fontsize=13)
	fig.text(0.5, 0.04, r'Log relative error between $FF_T$ and $FF_T^*$', ha='center', fontsize=13)
	num_bins=20
	rwidth = 1
	x_range = (-3,2)

	ax[0].set_title('With TATA', fontweight='bold')
	ax[0].grid(axis='y', alpha=0.75, color='gray', linestyle=':')
	ax[0].hist(ff_error[has_tata_inds], bins=num_bins, rwidth=rwidth, range=x_range, density=True)

	ax[1].set_title('Without TATA', fontweight='bold')
	ax[1].grid(axis='y', alpha=0.75, color='gray', linestyle=':')
	ax[1].hist(ff_error[no_tata_inds], bins=num_bins, rwidth=rwidth, range=x_range, density=True)

	# Calculate the point-biserial correlation
	M = np.mean(ff_error)
	n = len(ff_error)
	n1 = len(has_tata_inds)
	n0 = len(no_tata_inds)
	S = np.sqrt(1/(n-1))*sum((ff_error-M)**2)
	M1 = np.mean([ff_error[i] for i in has_tata_inds])
	M0 = np.mean([ff_error[i] for i in no_tata_inds])
	r = ((M1-M0)/S)*np.sqrt(n1*n0/(n*(n-1)))

	print("Point-biserial correlation: ", r)

	fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

	plt.savefig('TATA_density.svg')
	plt.show()

def plot_burstiness_range(num, ksyn, koff, kon, alpha, beta, time, burstiness, seed=None):
	'''Plots a series of SSA simulations for genes in the Larsson dataset ordered by whichever measure of burstiness
	is passed as the burstiness parameter to the function.'''
	import SSA_telegraph as ssa
	bursty_arr = burstiness(alpha, beta, ksyn, koff, kon)
	dtype = [('burstiness', float),('k_0', float),('k_off',float),('k_on',float),('alpha',float),('beta',float)]
	sort = np.sort(np.array(list(zip(bursty_arr, ksyn, koff, kon, alpha, beta)), dtype=dtype), axis=0, order='burstiness')
	delta_i = int(np.floor(len(sort)/num))
	figs = [None]*num
	axs = [None]*num
	for i in range(num):
		burstiness = sort[i*delta_i][0]
		k0 = sort[i*delta_i][1]
		k1 = 1
		koff = sort[i*delta_i][2]
		kon = sort[i*delta_i][3]
		alpha = sort[i*delta_i][4]
		beta = sort[i*delta_i][5]
		title = r'Burstiness=' + r'{:.4f}'.format(burstiness) + r', $\alpha$=' + r'{:.2f}'.format(alpha) + r', $\beta=$' + r'{:.2f}'.format(beta)
		ssa.plot(1, 0, None, k0, 1, kon, koff, 1, plt_mean_variance=False, plt_analytical=False, plt_samples=True, save=True, time=time, title=title, y_lim=75, seed=seed, fontsize=19, axes=False)


### UNCOMMENT TO GENERATE FIGURES

## Define paths to datasets

## path to Larsson et. al. data
#L_path = r'burstdata.csv'

## path to ontology data
#O_path = r'Gene_Parameter_Data.csv'

## Import data sets
#L_data = pd.read_csv(L_path)
#O_data = pd.read_csv(O_path)

## Get parameter arrays
#kon, koff, ksyn, gene_names = get_ks(L_data)
#alpha, beta, ff = get_alpha_beta_ff(kon, koff, ksyn)

## Generate figure 8
#plot_ff_ffstar_relerror(alpha, ff)

## Generate figure 9a
#TATA_ff_rel_error_correlation(alpha, ff, O_data, gene_names)

## Generate figure 9b
#gene_length_against_ff_rel_error(alpha, ff, O_data, gene_names)

## Define burstiness measures
#cv_fpt_sq = lambda alpha, beta, ksyn, koff, kon: 1+((1-beta)*2*alpha**2*beta**3)/((alpha*beta**2+1)**2)
#ff_copy_num = lambda alpha, beta, ksyn, koff, kon: 1+beta**2*alpha/(1+beta/koff)

## Generate a superset of those plots in figure 10
#plot_burstiness_range(20, ksyn, koff, kon, alpha, beta, 120, cv_fpt_sq, seed=1234)

## Generate a superset of those plots in figure 11
#plot_burstiness_range(20, ksyn, koff, kon, alpha, beta, 120, ff_copy_num, seed=1234)
