import numpy as np

from fanova import fANOVA
import fanova.visualizer

import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import os
path = os.path.dirname(os.path.realpath(__file__))

# directory in which you can find all plots
plot_dir = path + '/example_data/test_plots'

# artificial dataset (here: features)
features = np.loadtxt(path + '/example_data/diabetes_features.csv', delimiter=",")
responses = np.loadtxt(path + '/example_data/diabetes_responses.csv', delimiter=",")

# config space
pcs = list(zip(np.min(features,axis=0), np.max(features, axis=0)))
cs = ConfigSpace.ConfigurationSpace()
for i in range(len(pcs)):
	cs.add_hyperparameter(UniformFloatHyperparameter("%i" %i, pcs[i][0], pcs[i][1]))


# create an instance of fanova with trained forest and ConfigSpace
f = fANOVA(X = features, Y = responses, config_space=cs)

# marginal of particular parameter:
dims = (1, )
res = f.quantify_importance(dims)
print(res)

# getting the 10 most important pairwise marginals sorted by importance
best_margs = f.get_most_important_pairwise_marginals(n=10)
print(best_margs)

# visualizations:
# first create an instance of the visualizer with fanova object and configspace
vis = fanova.visualizer.Visualizer(f, cs, 'example_output')
# creating the plot of pairwise marginal:
vis.plot_pairwise_marginal((0,2), resolution=20)
# creating all plots in the directory
vis.create_all_plots(plot_dir)
