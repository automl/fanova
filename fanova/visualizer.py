from ConfigSpace.hyperparameters import CategoricalHyperparameter
import os
import numpy as np
import warnings
import pickle
import re
import logging

import matplotlib.pyplot as plt
import itertools as it
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Visualizer(object):

    def __init__(self, fanova, cs, directory, y_label='Performance'):
        """
        Parameters
        ------------
        fanova: fANOVA object

        cs : ConfigSpace instantiation

        directory: str
            Path to the directory in which all plots will be stored
        """
        self.fanova = fanova
        self.cs = cs
        self.cs_params = cs.get_hyperparameters()
        assert os.path.exists(directory), "directory %s doesn't exist" % directory
        self.directory = directory
        self._y_label = y_label
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

    def create_all_plots(self, three_d=True, **kwargs):
        """
        Creates plots for all main effects and stores them into a directory
        Specifically, all single and pairwise marginals are plotted

        Parameters
        ----------
        three_d: boolean
            whether or not to plot pairwise marginals in 3D-plot
        """
        # single marginals
        for param_idx in range(len(self.cs_params)):
            param_name = self.cs_params[param_idx].name
            plt.close()
            outfile_name = os.path.join(self.directory, param_name.replace(os.sep, "_") + ".png")
            self.logger.info("creating %s" % outfile_name)

            self.plot_marginal(param, show=False, **kwargs)
            plt.savefig(outfile_name)

        # additional pairwise plots:
        dimensions = list(range(len(self.cs_params)))
        combis = list(it.combinations(dimensions, 2))
        for combi in combis:
            param_names = [self.cs_params[p].name for p in combi]
            plt.close()
            param_names = str(param_names)
            param_names = re.sub('[!,@#\'\n$\[\]]', '', param_names)
            outfile_name = os.path.join(self.directory, str(param_names).replace(" ", "_") + ".png")
            self.logger.info("creating %s" % outfile_name)
            self.plot_pairwise_marginal(combi, three_d=three_d, **kwargs)
            plt.savefig(outfile_name)

    def generate_pairwise_marginal(self, param_indices, resolution=20):
        """
        Creates a plot of pairwise marginal of a selected parameters

        Parameters
        ------------
        param_list: list of ints or strings
            Contains the selected parameters

        resolution: int
            Number of samples to generate from the parameter range as
            values to predict
        """
        assert len(param_indices) == 2, "You have to specify 2 (different) parameters"
        grid_list, param_names = [], []
        if (isinstance(self.cs_params[param_indices[0]], (CategoricalHyperparameter)) or
            isinstance(self.cs_params[param_indices[1]], (CategoricalHyperparameter))):
            choice_arr = []
            param_names = []
            choice_vals = []
            for p in param_indices:
                if isinstance(self.cs_params[p], (CategoricalHyperparameter)):
                    choice_arr.append(self.cs_params[p].choices)
                    choice_vals.append(np.arange(len(self.cs_params[p].choices)))
                else:
                    lower_bound = self.cs_params[p].lower
                    upper_bound = self.cs_params[p].upper
                    grid = np.linspace(lower_bound, upper_bound, resolution)
                    choice_arr.append(grid)
                    choice_vals.append(grid)

                param_names.append(self.cs_params[p].name)

            choice_arr = [[choice_arr[1], choice_arr[0]] if len(choice_arr[1]) > len(choice_arr[0]) else [choice_arr[0], choice_arr[1]]]
            choice_vals = [[choice_vals[1], choice_vals[0]] if len(choice_vals[1]) < len(choice_vals[0]) else [choice_vals[0], choice_vals[1]]]
            choice_arr = np.asarray(choice_arr).squeeze()
            choice_vals = np.asarray(choice_vals).squeeze()
            param_indices = [[param_indices[1], param_indices[0]] if len(choice_vals[1]) < len(choice_vals[0]) else [param_indices[0], param_indices[1]]]
            choice_arr = np.asarray(choice_arr).squeeze()
            choice_vals = np.asarray(choice_vals).squeeze()
            param_indices = np.asarray(param_indices).squeeze()
            zz = np.zeros((len(choice_vals[0]), len(choice_vals[1])))

            for i, x_value in enumerate(choice_vals[0]):
                for j, y_value in enumerate(choice_vals[1]):
                    zz[i][j] = self.fanova.marginal_mean_variance_for_values(param_indices, [x_value, y_value])[0]

            return choice_arr, zz

        else:

            for p in param_indices:
                lower_bound = self.cs_params[p].lower
                upper_bound = self.cs_params[p].upper
                param_names.append(self.cs_params[p].name)
                grid = np.linspace(lower_bound, upper_bound, resolution)
                grid_list.append(grid)

            zz = np.zeros([resolution * resolution])
            for i, y_value in enumerate(grid_list[1]):
                for j, x_value in enumerate(grid_list[0]):
                    zz[i * resolution + j] = self.fanova.marginal_mean_variance_for_values(param_indices, [x_value, y_value])[0]

            zz = np.reshape(zz, [resolution, resolution])

            return grid_list, zz

    def plot_pairwise_marginal(self, param_list, resolution=20, show=False, three_d=True, colormap=cm.jet,
                               add_colorbar=True):
        """
        Creates a plot of pairwise marginal of a selected parameters

        Parameters
        ------------
        param_list: list of ints or strings
            Contains the selected parameters

        resolution: int
            Number of samples to generate from the parameter range as
            values to predict

        show: boolean
            whether to call plt.show() to show plot directly as interactive matplotlib-plot

        three_d: boolean
            whether or not to plot pairwise marginals in 3D-plot

        colormap: matplotlib.Colormap
            which colormap to use for the 3D plots

        add_colorbar: bool
            whether to add the colorbar for 3d plots
        """
        assert len(param_list) == 2, "You have to specify 2 (different) parameters"
        param_names, param_indices= [], []

        for p_idx in param_list:
            if type(p_idx) == str:  # if param_list consists of parameter names
                p_idx = self.cs.get_idx_by_hyperparameter_name(p_idx)
            param_names.append(self.cs_params[p_idx].name)
            param_indices.append(p_idx)

        first_is_cat = isinstance(self.cs_params[param_indices[0]], CategoricalHyperparameter)
        second_is_cat = isinstance(self.cs_params[param_indices[1]], CategoricalHyperparameter)

        fig = plt.figure()
        plt.title('%s and %s' % (param_names[0], param_names[1]))

        if first_is_cat or second_is_cat:
            # At least one of the two parameters is categorical
            choices, zz = self.generate_pairwise_marginal(param_indices, resolution)
            if first_is_cat and second_is_cat:
                # Both parameters are categorical -> create hotmap
                plt.imshow(zz, cmap='hot', interpolation='nearest')
                plt.xticks(np.arange(0, len(choices[0])), choices[0], fontsize=8)
                plt.yticks(np.arange(0, len(choices[1])), choices[1], fontsize=8)
                plt.xlabel(param_names[0])
                plt.ylabel(param_names[1])
                plt.colorbar().set_label(self._y_label)
            else:
                # Only one of them is categorical -> create multi-line-plot
                cat_choices = self.cs_params[param_indices[0]].choices if first_is_cat else self.cs_params[param_indices[1]].choices

                for i, cat in enumerate(cat_choices):
                    plt.plot(zz[i], label='%s' % str(cat))

                plt.ylabel(self._y_label)
                plt.xlabel(param_names[0] if second_is_cat else param_names[1])  # x-axis displays non-categorical
                plt.legend()
                plt.tight_layout()

        else:
            # No categoricals -> create 3D-plot
            grid_list, zz = self.generate_pairwise_marginal(param_indices, resolution)

            z_min, z_max = zz.min(), zz.max()
            display_xx, display_yy = np.meshgrid(grid_list[0], grid_list[1])

            if three_d:
                ax = Axes3D(fig)
                surface = ax.plot_surface(display_xx, display_yy, zz,
                                          rstride=1, cstride=1, cmap=colormap, linewidth=0, antialiased=False)
                ax.set_xlabel(param_names[0])
                ax.set_ylabel(param_names[1])
                ax.set_zlabel(self._y_label)
                if add_colorbar:
                    fig.colorbar(surface, shrink=0.5, aspect=5)

            else:
                plt.pcolor(display_xx, display_yy, zz, cmap=colormap, vmin=z_min, vmax=z_max)
                plt.xlabel(param_names[0])

                if self.cs_params[param_indices[0]].log:
                    plt.xscale('log')
                if self.cs_params[param_indices[1]].log:
                    plt.yscale('log')

                plt.ylabel(param_names[1])
                plt.colorbar()

        if show:
            plt.show()
        else:
            interact_dir = os.path.join(self.directory, 'interactive_plots')
            if not os.path.exists(interact_dir):
                self.logger.info('creating %s' % interact_dir)
                os.makedirs(interact_dir)
            try:
                pickle.dump(fig, open(interact_dir + '/%s_%s.fig.pkl' % (param_names[0], param_names[1]), 'wb'))
            except AttributeError as err:
                self.logger.debug(err, exc_info=1)
                self.logger.info("Pickling the interactive pairwise-marginal plot (%s) raised an exception. Resume "
                                 "without pickling. ", str(param_names))

        return plt

    def generate_marginal(self, param, resolution=100):
        """
        Creates marginals of a selected parameter for own plots

        Parameters
        ------------
        param: int or str
            Index of chosen parameter in the ConfigSpace (starts with 0)

        resolution: int
            Number of samples to generate from the parameter range as
            values to predict

        """
        if type(param) == str:
            param = self.cs.get_idx_by_hyperparameter_name(param)

        if isinstance(self.cs_params[param], (CategoricalHyperparameter)):
            param_name = self.cs_params[param].name
            labels= self.cs_params[param].choices
            categorical_size  = len(self.cs_params[param].choices)
            marginals = [self.fanova.marginal_mean_variance_for_values([param], [i]) for i in range(categorical_size)]
            mean, v = list(zip(*marginals))
            std = np.sqrt(v)
            return mean, std

        else:
            lower_bound = self.cs_params[param].lower
            upper_bound = self.cs_params[param].upper
            log = self.cs_params[param].log
            if log:
                # JvR: my conjecture is that ConfigSpace uses the natural logarithm
                base = np.e
                log_lower = np.log(lower_bound) / np.log(base)
                log_upper = np.log(upper_bound) / np.log(base)
                grid = np.logspace(log_lower, log_upper, resolution, endpoint=True, base=base)
                '''
                if abs(grid[0] - lower_bound) > 0.00001:
                    raise ValueError()
                if abs(grid[-1] - upper_bound) > 0.00001:
                    raise ValueError()
                '''
            else:
                grid = np.linspace(lower_bound, upper_bound, resolution)
            mean = np.zeros(resolution)
            std = np.zeros(resolution)

            dim = [param]
            for i in range(0, resolution):
                (m, v) = self.fanova.marginal_mean_variance_for_values(dim, [grid[i]])
                mean[i] = m
                std[i] = np.sqrt(v)
            return mean, std, grid

    def plot_marginal(self, param, resolution=100, log_scale=None, show=True,
                      incumbents=None):
        """
        Creates a plot of marginal of a selected parameter

        Parameters
        ------------
        param: int or str
            Index of chosen parameter in the ConfigSpace (starts with 0)

        resolution: int
            Number of samples to generate from the parameter range as
            values to predict

        log_scale: boolean
            If log scale is required or not. If no value is given, it is
            deduced from the ConfigSpace provided

        show: boolean
            whether to call plt.show() to show plot directly as interactive matplotlib-plot

        incumbents: List[Configuration]
            list of ConfigSpace.Configurations that are marked as incumbents
        """
        param_idx = param
        if type(param_idx) == str:
            param_idx = self.cs.get_idx_by_hyperparameter_name(param_idx)
        param_name = self.cs_params[param_idx].name

        # check if categorical
        if isinstance(self.cs_params[param_idx], (CategoricalHyperparameter)):
            # PREPROCESS
            labels = self.cs_params[param_idx].choices
            categorical_size = len(self.cs_params[param_idx].choices)
            indices = np.arange(1, categorical_size+1, 1)
            mean, std = self.generate_marginal(param_idx)
            min_y = mean[0]
            max_y = mean[0]

            # PLOT
            b = plt.boxplot([[x] for x in mean])
            plt.xticks(indices, labels)
            # blow up boxes
            for box, std_ in zip(b["boxes"], std):
                y = box.get_ydata()
                y[2:4] = y[2:4] + std_
                y[0:2] = y[0:2] - std_
                y[4] = y[4] - std_
                box.set_ydata(y)
                min_y = min(min_y, y[0] - std_)
                max_y = max(max_y, y[2] + std_)

            plt.ylim([min_y, max_y])

            plt.ylabel(self._y_label)
            plt.xlabel(param_name)
            plt.tight_layout()

        else:

            # PREPROCESS
            mean, std, grid = self.generate_marginal(param_idx, resolution)
            mean = np.asarray(mean)
            std = np.asarray(std)

            lower_curve = mean - std
            upper_curve = mean + std

            if log_scale is None:
                log_scale = self.cs_params[param].log or (np.diff(grid).std() > 0.000001)

            # PLOT
            if log_scale:
                if (np.diff(grid).std() > 0.000001):
                    self.logger.info("It might be better to plot this parameter"
                                     " '%s' in log-scale.", param_name)
                plt.semilogx(grid, mean, 'b', label='predicted %s' % self._y_label)
            else:
                plt.plot(grid, mean, 'b', label='predicted %s' % self._y_label)
            plt.fill_between(grid, upper_curve, lower_curve, facecolor='red',
                             alpha=0.6, label='std')

            if incumbents is not None:
                if not isinstance(incumbents, list):
                    incumbents = [incumbents]
                values = [inc[param_name] for inc in incumbents if param_name in inc and inc[param_name] is not None]
                indices = [(np.abs(np.asarray(grid) - val)).argmin() for val in values]
                if len(indices) > 0:
                    plt.scatter(list([grid[idx] for idx in indices]),
                                list([mean[idx] for idx in indices]),
                                label='incumbent', c='black', marker='.', zorder=999)

            plt.xlabel(param_name)
            plt.ylabel(self._y_label)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

        if show:
            plt.show()
        else:
            return plt

    def create_most_important_pairwise_marginal_plots(self, params=None, n=20, three_d=True):
        """
        Creates plots of the n most important pairwise marginals of the whole ConfigSpace

        Parameters
        ------------
        params: list
             Contains the selected parameters for pairwise evaluation
        n: int
             The number of most relevant pairwise marginals that will be returned
        three_d: boolean
            whether or not to plot pairwise marginals in 3D-plot

        """
        if self.fanova._dict:
            most_important_pairwise_marginals = self.fanova.tot_imp_dict
        else:
            if params is not None:
                most_important_pairwise_marginals = self.fanova.get_most_important_pairwise_marginals(params=params)
            else:
                most_important_pairwise_marginals = self.fanova.get_most_important_pairwise_marginals(n=n)

        for param1, param2 in most_important_pairwise_marginals:
            param1, param2 = self.cs.get_idx_by_hyperparameter_name(param1), self.cs.get_idx_by_hyperparameter_name(param2)
            param_names = [self.cs_params[param1].name, self.cs_params[param2].name]
            param_names = str(param_names)
            param_names = re.sub('[!,@#\'\n$\[\]]', '', param_names)
            outfile_name = os.path.join(self.directory, str(param_names).replace(" ","_") + ".png")
            self.logger.info("creating %s" % outfile_name)
            self.plot_pairwise_marginal((param1, param2), show=False, three_d=three_d)
            plt.savefig(outfile_name)

