import papermill as pm
from itertools import product
import sys
import os


template_dir = '~/Documents/GitHub/StructuredPredictionError/papermill_templates/'
wdir = '~/Documents/GitHub/StructuredPredictionError/papermill_templates/output/'

rl_dict = {
	'niter': [100],
	'n': [400],
	'p': [500],
	's': [30],
	'k': [5,10],
	'delta':[0.5,0.8],
	'alpha':[1.],
	'lambd':[1.],
	'noise_kernel': ['matern'],
	'noise_length_scale':[1.],
	'noise_nu':[0.5, 2.5],
	'X_kernel': ['rbf'],
	'X_length_scale':[5.],
	'X_nu':[1.],
}


rf_dict = {
	'niter': [100],
	'n': [900],
	'p': [30],
	's': [30],
	'k': [5,10],
	'delta':[0.5,0.8],
	'n_estimators': [100],
	'max_depth': [6],
	'noise_kernel': ['matern'],
	'noise_length_scale':[1.],
	'noise_nu':[0.5, 2.5],
	'X_kernel': ['rbf'],
	'X_length_scale':[5.],
	'X_nu':[1.],
}

rfrefits_dict = rf_dict


def product_dict(**kwargs):
	keys = kwargs.keys()
	vals = kwargs.values()
	for instance in product(*vals):
		yield dict(zip(keys, instance))


def run_nbs(model, folder_name):
	if model == 'RL':
		params = product_dict(**rl_dict)
		# input_nb = 'RLTemplate.ipynb'
		# indir = wdir + 'RL/exp_kern_multik/'
	elif model == 'RF':
		params = product_dict(**rf_dict)
		# input_nb = 'RFTemplate.ipynb'
		# indir = wdir + 'RF/exp_kern_multik/'
	elif model == 'RFRefits':
		params = product_dict(**rfrefits_dict)
		# input_nb = 'RFRefitsTemplate.ipynb'
		# indir = wdir + 'RFRefits/exp_kern_multik/'

	input_nb = model + "Template.ipynb"
	indir = wdir + model + '/' + folder_name + '/'

	inputfp = os.path.expanduser(template_dir) + input_nb
	if not os.path.exists(inputfp):
		os.makedirs(inputfp)
	
	output_nb = 'output.ipynb'
	
	for i,param in enumerate(params):
		outputdir = os.path.expanduser(indir) + 'run_' + str(i) + '/'
		outputfp = outputdir + output_nb
		if not os.path.exists(outputdir):
			os.makedirs(outputdir)
		param['savedir'] = outputdir
		param['idx'] = i
		pm.execute_notebook(
			inputfp,
			outputfp,
			parameters=param
		)


if __name__ == "__main__":
	run_nbs(sys.argv[1], sys.argv[2])







