##File to plot the long term plact stuff
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import time, warnings
import sys,os
from collections import defaultdict
from itertools import product
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


import group_explanations as ge
from learning_env import eval_formula
from learning_env import dec_rule_env
import act_exp_io as aeio
import cost_funcs as cf
import linear_models as lm
import segregation_index as si
import long_term_impact as lti

sys.path.insert(0, "../util/")
import output as out
from datasets import file_util as fu
from persistence import load_params, save_params

import experiment as exp

kmeans_param_grid = {'n_clusters':list(range(1,101))}

class UtilityThresholds:

    def __init__(self):
        self.dataset, self.models_other_than_rules = exp.base_exp(return_vars=True)
        self.prediction_task = exp.dataset_info[self.dataset]['prediction_task']
        self.res_dir = 'results/{}'.format(self.dataset)
        out.create_dir(self.res_dir)
        self.res_file_path = self.res_dir + '/res_utilities_thresholds.txt'
        self.seg_file_path = self.res_dir + '/res_segregation.txt'
        self.wiki_parent_path = "Actionable-Explanations/Simple-Explanations-{}".format(self.dataset)
        self.sens_group_desc = exp.dataset_info[self.dataset]['sens_f']
        self.cost_groups = {cf.ONE_GROUP_IND: "all", 0: self.sens_group_desc[0], 1: self.sens_group_desc[1]}
        self.segregation_indices = [si.Atkinson, si.Centralization, si.Clustering]

    def find_best_params(self, X, Y, param_grid, eval_func, learning_algo):
        # works only for sklearn.clustering algos
        best_param, best_eval = None, None
        for perm in product(*param_grid.values()):
            param = dict(zip(param_grid.keys(), perm))
            alg = learning_algo(**param, random_state=42, n_jobs=-1)
            alg.fit(X)
            eval_ = eval_func(alg.labels_, Y)
            if best_eval is None or best_eval < eval_:
                best_eval, best_param = eval_, param
        return best_param

    def find_c_ij(self, X, Y, i, j, centroids):
        assert centroids.shape[0] == len(np.unique(Y))
        if i==j:
            return 1
        X_i, X_j = X[Y == i], X[Y == j]
        centroid_i, centroid_j = np.mean(X_i, axis=0), np.mean(X_j, axis=0)
        print (np.count_nonzero(np.all(np.isclose(centroid_i, centroids, rtol=0, atol=0.01), axis=1)), 
            np.count_nonzero(np.all(np.isclose(centroid_j, centroids, rtol=0, atol=0.01), axis=1)))
        assert np.count_nonzero(np.all(np.isclose(centroid_i, centroids, rtol=0, atol=0.01), axis=1)) > 0 and np.count_nonzero(np.all(np.isclose(centroid_j, centroids, rtol=0, atol=0.01), axis=1)) > 0
        d_ij = euclidean(centroid_i, centroid_j)
        return np.exp(-d_ij)

    def find_neighbourhoods(self, X, Y, tau_sens, tau_nosens, model=None):
        out.create_dir('./params')
        k_means_params = aeio.load_params('./params/KMeans', '{}_{}_{}'.format(model.filename(), 
            tau_sens, tau_nosens) if model is not None else '{}_{}'.format(tau_sens, tau_nosens))
        if k_means_params is None:
            k_means_params = self.find_best_params(X, Y, kmeans_param_grid, adjusted_mutual_info_score, KMeans)
            aeio.save_params('./params/KMeans', '{}_{}_{}'.format(model.filename(), tau_sens, tau_nosens), k_means_params)
        alg = KMeans(**k_means_params, n_jobs=-1, random_state=42)
        alg.fit(X)
        return alg.cluster_centers_, alg.labels_, k_means_params

    def abs_clustering_index(self, X, Y, sens_group, tau_sens, tau_nosens, model=None):
        # Formula from https://www.census.gov/hhes/www/housing/resseg/pdf/app_b.pdf
        cluster_centers, labels, k_means_params = self.find_neighbourhoods(X, Y, tau_sens, tau_nosens, model)
        print (cluster_centers)
        if self.prev_labels is None:
            self.prev_labels = labels
        else:
            print (np.where(self.prev_labels != labels)[0])
            print (self.prev_labels[np.where(self.prev_labels != labels)[0]], labels[np.where(self.prev_labels != labels)[0]])
            print (sens_group[np.where(self.prev_labels != labels)[0]])
        num_classes = len(np.unique(labels))
        num_sens, num_nosens = np.count_nonzero(sens_group), np.count_nonzero(~sens_group)
        a_00, a_01, a_10, a_11 = 0, 0, 0, 0
        for i in range(num_classes):
            num_sens_i = np.count_nonzero(np.logical_and(labels == i, sens_group))
            b_00, b_10 = 0, 0
            for j in range(num_classes):
                num_sens_j = np.count_nonzero(np.logical_and(labels == j, sens_group))
                size_of_class_j = np.count_nonzero(labels == j)
                c_ij = self.find_c_ij(X, labels, i, j, centroids=cluster_centers)
                a_01 += c_ij
                a_11 += c_ij
                b_00 += c_ij * num_sens_j
                b_10 += c_ij * size_of_class_j
            a_00 += (num_sens_i/num_sens) * b_00
            a_10 += (num_sens_i/num_sens) * b_10
        a_01 += (num_sens/(num_classes**2)) * a_01
        a_11 += (num_sens/(num_classes**2)) * a_11
        return (a_00 - a_01) / (a_10 - a_11), k_means_params
        # return a_00 / a_10 # non-normalized value

    def gini_index(self, X, Y, sens_group, tau_sens, tau_nosens, model=None):
        # Formula from https://www.census.gov/hhes/www/housing/resseg/pdf/app_b.pdf
        cluster_centers, labels, k_means_params = self.find_neighbourhoods(X, Y, tau_sens, tau_nosens, model)
        if self.prev_labels is None:
            self.prev_labels = labels
        else:
            print (np.where(self.prev_labels != labels)[0])
            print (self.prev_labels[np.where(self.prev_labels != labels)[0]], labels[np.where(self.prev_labels != labels)[0]])
            print (sens_group[np.where(self.prev_labels != labels)[0]])
        # print (cluster_centers)
        n = len(np.unique(labels))
        num_sens, T = np.count_nonzero(sens_group), len(Y)
        P = num_sens/T
        numerator, denominator = 0, (2*T*T*P*(1-P))
        for i in range(n):
            x_i = np.count_nonzero(np.logical_and(labels == i, sens_group))
            t_i = np.count_nonzero(labels == i)
            p_i = x_i/t_i
            for j in range(n):
                x_j = np.count_nonzero(np.logical_and(labels == j, sens_group))
                t_j = np.count_nonzero(labels == j)
                p_j = x_j/t_j
                print ("i={}, j={}, x_i = {}, t_i = {}, p_i = {}, x_j = {}, t_j = {}, p_j = {} ".format(
                    i, j, x_i, t_i, p_i, x_j, t_j, p_j))
                numerator += t_i * t_j * abs(p_i - p_j)
        print (numerator/denominator)
        print ("\n\n")
        return (numerator/denominator), k_means_params

    def plot_data_points(self, X, sens_group, tau_sens, tau_nosens, format='pdf'):
        tsne = TSNE(n_components=2, random_state=42, n_iter=250)
        X_new = tsne.fit_transform(X)
        X_new_sens, X_new_nosens = X_new[sens_group], X_new[~sens_group]
        filename = "scatter_{}_{}".format(tau_sens, tau_nosens)
        figpath = self.res_dir + "/disparity_plots/" + filename + '.' + format
        if not os.path.exists(self.res_dir + "/disparity_plots/"):
            os.mkdir(self.res_dir + "/disparity_plots/")
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.scatter(X_new_sens[:,0], X_new_sens[:,1], color='blue', marker='o', label='Sens')
        ax.scatter(X_new_nosens[:,0], X_new_nosens[:,1], color='red', marker='x', label='Non-Sens')
        ax.set_title("Tau sens: {}, Tau non-sens: {}".format(tau_sens, tau_nosens))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(figpath, format=format, bbox_inches='tight')
        plt.clf()
        plt.close()

    def initialize_variables(self, model, index_name='gini_index'):
        ## change index_name and self.gini_index to the appropriate function
        self.index_name = index_name
        self.feature_info = model.feature_info
        self.x_control = model.x_control
        self.x_test_original, self.x_train_original = model.x_test, model.x_train
        scaler = MinMaxScaler()
        scaler.fit(self.x_train_original)
        self.x_train = scaler.transform(self.x_train_original)
        self.x_test = scaler.transform(self.x_test_original)
        self.y_test, self.y_train = (model.y_test).astype(bool if self.prediction_task == exp.CLASSIFICATION else float), (model.y_train).astype(bool if self.prediction_task == exp.CLASSIFICATION else float)

        self.sens_group = ~model.x_control[self.sens_group_desc[-1]]
        self.sens_group_train = ~model.x_control_train[self.sens_group_desc[-1]]
        self.sens_group_test = ~model.x_control_test[self.sens_group_desc[-1]]

        self.disparity_table_heading = ["Model", "Threshold for sens_group"]
        self.disparity_table_formats = [None, None]
        self.disparity_table_values = []
        self.disparity_table_values_old = []
        self.seg_index_mapping = {}

        self.taus_for_sens = lti.LongTermImpact.tau_sens
        # self.taus_for_sens = [0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04, 0.0425, 0.045, 0.0475]
        # self.taus_for_nosens = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 
        #     1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        self.taus_for_nosens = lti.LongTermImpact.tau_nosens[:1]

    def get_flipped_dataset(self, model, string):
        try:
            x = np.loadtxt("./flipped_datasets/{}/{}_all_x_{}.txt".format(self.dataset, model.filename(), string)).astype(float)
            y = np.loadtxt("./flipped_datasets/{}/{}_all_y_{}.txt".format(self.dataset, model.filename(), string)).astype(bool if self.prediction_task == exp.CLASSIFICATION else float)
            return x, y
        except:
            raise ValueError("First run experiment.py to generate {}_x_{}.txt".format(model.filename(), string))

    def get_double_flipped_utilities(self, model, threshold_sens, threshold_nosens):
        try:
            utilities = np.loadtxt("./flipped_datasets/{}/{}_new_utilities_test_{}_{}.txt".format(self.dataset, model.filename(), threshold_sens, threshold_nosens)).astype(float)
            return utilities
        except:
            warnings.warn("First run long_term_impact.py to generate {}_new_utilities_test_{}_{}.txt".format(model.filename(), threshold_sens, threshold_nosens))
            return None
            # raise ValueError("First run long_term_impact.py to generate {}_new_utilities_test_{}.txt".format(model.filename(), threshold))

    def get_utilities(self, model, string):
        try:
            utilities = np.loadtxt("./flipped_datasets/{}/{}_utilities_{}.txt".format(self.dataset, model.filename(), string)).astype(float)
            return utilities
        except:
            raise ValueError("First run experiment.py to generate {}_utilities_{}.txt".format(model.filename(), string))

    def get_anchor_indices(self, model, test_or_train):
        try:
            anchor_indices = np.loadtxt("./flipped_datasets/{}/{}_anchors_{}.txt".format(self.dataset, model.filename(), test_or_train))
            anchor_indices = anchor_indices[~np.isnan(anchor_indices)].astype(int)
            assert np.all(anchor_indices >= 0)
            return anchor_indices
        except:
            raise ValueError("First run experiment.py followed by long_term_impact.py to generate all anchor points")

    def run(self):
        learning_env = dec_rule_env.DecRuleEnv(self.dataset, self.sens_group_desc)
        learning_env.load_data(feature_engineering=True)
        self.initialize_variables(learning_env)

        if os.path.exists(self.res_dir + '/plots_pickled_data/seg_index_mapping.pkl' if not exp.FAIRNESS_CONSTRAINTS else self.res_dir + '/plots_pickled_data/seg_index_mapping_fc.pkl'):
            self.seg_index_mapping = joblib.load(self.res_dir + '/plots_pickled_data/seg_index_mapping.pkl' if not exp.FAIRNESS_CONSTRAINTS else self.res_dir + '/plots_pickled_data/seg_index_mapping_fc.pkl')
            seg_index_mapping_loaded = True
        else:
            seg_index_mapping_loaded = False

        all_models = self.models_other_than_rules
        with open(self.res_file_path, 'w') as group_res_file:
            group_res_file.write("= Disparity in effort analysis vs different utility thresholds =\n\n")
        with open(self.seg_file_path, 'w') as seg_res_file:
            seg_res_file.write("= Measuring Long Term Impact through Segregation =\n\n")

        for tau_nosens in self.taus_for_nosens:
            self.prev_labels = None
            self.disparity_table_values, self.disparity_table_values_old = [], []
            self.number_flipped_sens = {} # maping of model : number of people flipped
            self.number_flipped_nosens = {} # maping of model : number of people flipped
            self.new_abs_clustering_index = {} # mapping of model : list of new abs_clustering_index
            if not seg_index_mapping_loaded:
                self.seg_index_mapping[tau_nosens] = {k(self.sens_group_train, self.feature_info):{} for k in self.segregation_indices} # mapping of seg_index: {model : {'Original Population': index_val, 'tau_1': index_val, .....}}
            self.new_abs_clustering_index_params = {} # mapping of model : list of best params for clustering to find neighbourhoods
            self.old_abs_clustering_index = {} # mapping of model : initial abs_clustering index
            self.data_for_pdf = {} # mapping of model : dictionary (see below for details of this dictionary)
            with open(self.res_file_path, 'a') as group_res_file:
                group_res_file.write("== Utility threshold for non sensitive people = {:.2f} ==\n\n".format(tau_nosens))
            for model in all_models:
                ### set values in dicts
                # self.old_abs_clustering_index[model], initial_population_params = self.gini_index(self.x_train, self.y_train, self.sens_group_train, 'inf', 'inf')
                # self.old_ssi[model], self.new_ssi[model] = si.ssi(self.x_train, self.y_train, self.sens_group_train, 'inf', 'inf'), []
                self.number_flipped_sens[model], self.number_flipped_nosens[model] = [], []
                self.new_abs_clustering_index[model] = []
                self.new_abs_clustering_index_params[model] = []
                self.data_for_pdf[model] = {} # mapping of tau_sens : list of populations (x_train_new)

                all_flipped_x_test, all_flipped_y_test = self.get_flipped_dataset(model, 'test')
                all_flipped_x_train, all_flipped_y_train = self.get_flipped_dataset(model, 'train')
                utilities, utilities_train = self.get_utilities(model, 'test'), self.get_utilities(model, 'train')

                clf = model
                exists, loaded_clf = aeio.load_model(clf, self.dataset)
                if exists:
                    clf = loaded_clf
                    print ("Loaded {}...".format(str(clf)))
                else:
                    raise ValueError("Run experiment.py first")

                if isinstance(clf, lm.LinReg) or isinstance(clf, lm.LogReg):
                    with open(self.res_file_path, 'a') as group_res_file:
                        group_res_file.write("\n{}\n".format(out.get_table(**aeio.get_regression_weights(self.feature_info, clf))))
                elif isinstance(clf, lm.DTReg) or isinstance(clf, lm.DT):
                    with open(self.res_file_path, 'a') as group_res_file:
                        group_res_file.write("\n{}\n\n".format(aeio.plot_dtree(self.res_dir, clf, self.feature_info)))
                        group_res_file.write("{}\n\n".format(aeio.plot_covar_matrix(self.res_dir, self.x_train, self.feature_info)))

                users_preds, users_preds_train = clf.predict(self.x_test).astype(bool if self.prediction_task == exp.CLASSIFICATION else float), \
                    clf.predict(self.x_train).astype(bool if self.prediction_task == exp.CLASSIFICATION else float)

                cost_funcs, _ = exp.dataset_info[self.dataset]['cost_funcs'](self.feature_info, 
                    self.x_train, self.sens_group_train, exp.dataset_info[self.dataset]['variable_constraints'])
                cost_funcs_rev, _ = exp.dataset_info[self.dataset]['cost_funcs'](self.feature_info, 
                    self.x_train, self.sens_group_train, exp.dataset_info[self.dataset]['variable_constraints_rev'])
                
                if not seg_index_mapping_loaded:
                    for k,v in self.seg_index_mapping[tau_nosens].items():
                        v[model] = {'Original Population': k.val(X=self.x_train, y=self.y_train, cost_funcs=cost_funcs, cost_funcs_rev=cost_funcs_rev, 
                            anchor_indices=self.get_anchor_indices(model, 'train'), y_pred=users_preds_train)}

                sub_filter_sens, sub_filter_sens_train = np.zeros(len(self.x_test), dtype=bool), np.zeros(len(self.x_train), dtype=bool)
                sub_filter_nosens, sub_filter_nosens_train = np.zeros(len(self.x_test), dtype=bool), np.zeros(len(self.x_train), dtype=bool)
                sub_filter_sens[np.where(np.logical_and(self.sens_group_test, users_preds < self.y_test))[0]] = 1
                sub_filter_nosens[np.where(np.logical_and(~self.sens_group_test, users_preds < self.y_test))[0]] = 1
                sub_filter_sens_train[np.where(np.logical_and(self.sens_group_train, users_preds_train < self.y_train))[0]] = 1
                sub_filter_nosens_train[np.where(np.logical_and(~self.sens_group_train, users_preds_train < self.y_train))[0]] = 1
                print (set(list(utilities)))
                # explanations_given_sens = np.where(np.logical_and(~np.all(all_flipped_x_test == self.x_test, axis=1), self.sens_group_test))[0]
                # explanations_given_nosens = np.where(np.logical_and(~np.all(all_flipped_x_test == self.x_test, axis=1), ~self.sens_group_test))[0]
                with open(self.res_file_path, 'a') as group_res_file:
                    group_res_file.write(" * For {}, # test explanations given = {} ({} sens, {} non-sens)\n\n".format(str(model), 
                        len(utilities[sub_filter_sens]) + len(utilities[sub_filter_nosens]), len(utilities[sub_filter_sens]), len(utilities[sub_filter_nosens])))
                    group_res_file.write(" * For {}, # train explanations given = {} ({} sens, {} non-sens)\n\n".format(str(model), 
                        len(utilities_train[sub_filter_sens_train]) + len(utilities_train[sub_filter_nosens_train]), len(utilities_train[sub_filter_sens_train]), 
                        len(utilities_train[sub_filter_nosens_train])))
                for tau_sens in self.taus_for_sens:
                    sens_flipped, nonsens_flipped = (np.where(np.logical_and(utilities > tau_sens, self.sens_group_test))[0], 
                        np.where(np.logical_and(utilities > tau_nosens, ~self.sens_group_test))[0])
                    sens_flipped_train, nonsens_flipped_train = (np.where(np.logical_and(utilities_train > tau_sens, self.sens_group_train))[0], 
                        np.where(np.logical_and(utilities_train > tau_nosens, ~self.sens_group_train))[0])
                    sens_utility_old, nosens_utility_old = np.mean(utilities[sens_flipped]), np.mean(utilities[nonsens_flipped])
                    new_x_test, new_y_test = self.x_test.copy(), self.y_test.copy()
                    new_x_test[sens_flipped,:], new_y_test[sens_flipped] = all_flipped_x_test[sens_flipped,:], all_flipped_y_test[sens_flipped]
                    new_x_test[nonsens_flipped,:], new_y_test[nonsens_flipped] = all_flipped_x_test[nonsens_flipped,:], all_flipped_y_test[nonsens_flipped]
                    new_x_train, new_y_train = self.x_train.copy(), self.y_train.copy()
                    new_x_train[sens_flipped_train,:], new_y_train[sens_flipped_train] = all_flipped_x_train[sens_flipped_train,:], all_flipped_y_train[sens_flipped_train]
                    new_x_train[nonsens_flipped_train,:], new_y_train[nonsens_flipped_train] = all_flipped_x_train[nonsens_flipped_train,:], all_flipped_y_train[nonsens_flipped_train]
                    # Find the abs_clustering/gini index of new population
                    # index_val, clustering_params = self.gini_index(new_x_train, new_y_train, self.sens_group_train, 'inf', 'inf')
                    # self.new_abs_clustering_index[model].append(index_val)
                    # self.new_abs_clustering_index_params[model].append(clustering_params)
                    self.data_for_pdf[model][tau_sens] = new_x_train
                    # Plot data distribution in 2D
                    # self.plot_data_points(new_x_train, self.sens_group_train, tau_sens, tau_nosens)
                    try:
                        new_y_pred, new_y_pred_train = model.predict(new_x_test).astype(bool if self.prediction_task == exp.CLASSIFICATION else float), \
                            model.predict(new_x_train).astype(bool if self.prediction_task == exp.CLASSIFICATION else float)
                    except:
                        _, clf = aeio.load_model(model, self.dataset)
                        new_y_pred, new_y_pred_train = clf.predict(new_x_test).astype(bool if self.prediction_task == exp.CLASSIFICATION else float), \
                            clf.predict(new_x_train).astype(bool if self.prediction_task == exp.CLASSIFICATION else float)
                    
                    cost_funcs, _ = exp.dataset_info[self.dataset]['cost_funcs'](self.feature_info, 
                        new_x_train, self.sens_group_train, exp.dataset_info[self.dataset]['variable_constraints'])
                    cost_funcs_rev, _ = exp.dataset_info[self.dataset]['cost_funcs'](self.feature_info, 
                        new_x_train, self.sens_group_train, exp.dataset_info[self.dataset]['variable_constraints_rev'])

                    if not seg_index_mapping_loaded:
                        for k,v in self.seg_index_mapping[tau_nosens].items():
                            v[model]['{:.2f}'.format(tau_sens)] = k.val(X=new_x_train, y=new_y_train, cost_funcs=cost_funcs, 
                                cost_funcs_rev=cost_funcs_rev, anchor_indices=self.get_anchor_indices(model, 'train'), y_pred=new_y_pred_train)

                    with open(self.res_file_path, 'a') as group_res_file:
                        group_res_file.write(" * For {} test set, with sens tau = {:.2f}, # flipped users = {} ({} sens, {} non-sens)\n\n".format(str(model), tau_sens,
                            len(sens_flipped) + len(nonsens_flipped), len(sens_flipped), len(nonsens_flipped)))
                        group_res_file.write(" * For {} train set, with sens tau = {:.2f}, # flipped users = {} ({} sens, {} non-sens)\n\n".format(str(model), tau_sens,
                            len(sens_flipped_train) + len(nonsens_flipped_train), len(sens_flipped_train), len(nonsens_flipped_train)))

                    self.number_flipped_sens[model].append(len(sens_flipped_train)/np.count_nonzero(users_preds_train < self.y_train))
                    self.number_flipped_nosens[model].append(len(nonsens_flipped_train)/np.count_nonzero(users_preds_train < self.y_train))
                    double_flipped_utilities = self.get_double_flipped_utilities(model, tau_sens, tau_nosens)
                    if double_flipped_utilities is not None:
                        sens_flipped_new, nonsens_flipped_new = (np.where(np.logical_and(double_flipped_utilities != 0, self.sens_group_test))[0], 
                            np.where(np.logical_and(double_flipped_utilities != 0, ~self.sens_group_test))[0])
                        # sens_flipped_new, nonsens_flipped_new = (np.where(np.logical_and(double_flipped_utilities > tau_sens, self.sens_group_test))[0], 
                        #     np.where(np.logical_and(double_flipped_utilities > tau_nosens, ~self.sens_group_test))[0])
                        sens_utility_new, nosens_utility_new = np.mean(double_flipped_utilities[sens_flipped_new]), np.mean(double_flipped_utilities[nonsens_flipped_new])
                        if len(self.disparity_table_heading) <= 2:
                            heading, formats, values = eval_formula.get_disparity_measures(new_y_test, new_y_pred, self.sens_group_test, sens_utility_new if not np.isnan(sens_utility_new) else 0., 
                                nosens_utility_new if not np.isnan(nosens_utility_new) else 0., self.prediction_task, return_heading_and_formats=True)
                            self.disparity_table_heading += heading
                            self.disparity_table_formats += formats
                        else:
                            values = eval_formula.get_disparity_measures(new_y_test, new_y_pred, self.sens_group_test, sens_utility_new if not np.isnan(sens_utility_new) else 0., 
                                nosens_utility_new if not np.isnan(nosens_utility_new) else 0., self.prediction_task, return_heading_and_formats=False)
                        old_values = eval_formula.get_disparity_measures(self.y_test, users_preds, self.sens_group_test, sens_utility_old if not np.isnan(sens_utility_old) else 0., 
                                nosens_utility_old if not np.isnan(nosens_utility_old) else 0., self.prediction_task, return_heading_and_formats=False)
                        self.disparity_table_values.append([str(model), "{:.2f}".format(tau_sens)] + values)
                        self.disparity_table_values_old.append([str(model), "{:.2f}".format(tau_sens)] + old_values)

            # with open(self.res_file_path, 'a') as group_res_file:
            #     group_res_file.write("{}\n\n".format(out.get_table(self.disparity_table_heading, 
            #         self.disparity_table_values, val_format=self.disparity_table_formats)))
            #     for i in range(3, len(self.disparity_table_heading)):
            #         self.disparity_table_values = np.array(self.disparity_table_values)
            #         self.disparity_table_values_old = np.array(self.disparity_table_values_old)
            #         heading = '_'.join(self.disparity_table_heading[i].split("<<BR>>")[0].strip().lower().split(" "))
            #         group_res_file.write(aeio.get_utility_threshold_plots(self.res_dir, all_models, self.disparity_table_heading[:2] + [self.disparity_table_heading[i]], 
            #             np.append(self.disparity_table_values[:,:2], self.disparity_table_values[:,i:i+1], axis=1),
            #             np.append(self.disparity_table_values_old[:,:2], self.disparity_table_values_old[:,i:i+1], axis=1),
            #             tau_nosens, filename='utility_threshold_{}_{}'.format(tau_nosens, heading), plot_title=''))
                # group_res_file.write("\n{}\n\n".format(aeio.get_abs_clustering_plots(self.res_dir, self.taus_for_sens, self.number_flipped_sens, 
                #     self.number_flipped_nosens, tau_nosens, 'Fraction of Flipped Users', filename='number_of_users_flipped_{}'.format(tau_nosens), plot_title='')))
                # for wiki_path in aeio.get_pdf_plots(self.res_dir, self.x_train, self.sens_group_train, self.taus_for_sens, self.feature_info, self.data_for_pdf, tau_nosens):
                #     group_res_file.write("\n{}\n\n".format(wiki_path))

        out.create_dir(self.res_dir + '/plots_pickled_data')
        joblib.dump(self.seg_index_mapping, self.res_dir + '/plots_pickled_data/seg_index_mapping.pkl' if not exp.FAIRNESS_CONSTRAINTS else self.res_dir + '/plots_pickled_data/seg_index_mapping_fc.pkl')

        with open(self.seg_file_path, 'a') as seg_res_file:
            for wiki_path in aeio.get_segregation_plots_new(self.res_dir, self.seg_index_mapping, exp.FAIRNESS_CONSTRAINTS):
                seg_res_file.write("\n{}\n\n".format(wiki_path))

        out.upload_results([self.res_dir + '/disparity_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.png')
        out.upload_results([self.res_dir + '/pdf_before_after_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.png')
        out.upload_results([self.res_dir + '/segregation_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.png')
        out.upload_results([self.res_dir + '/disparity_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.pdf')
        out.upload_results([self.res_dir + '/pdf_before_after_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.pdf')
        out.upload_results([self.res_dir + '/segregation_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.pdf')

if __name__ == "__main__":
    ut = UtilityThresholds()
    ut.run()
