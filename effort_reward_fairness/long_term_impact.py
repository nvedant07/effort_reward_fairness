from multiprocessing import Process, Queue

import numpy as np
import time
import sys
from collections import defaultdict
from itertools import product
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

import group_explanations as ge
from learning_env import eval_formula
from learning_env import dec_rule_env
import act_exp_io as aeio
import cost_funcs as cf
import linear_models as lm

sys.path.insert(0, "../util/")
import output as out
from datasets import file_util as fu

import experiment as exp

class LongTermImpact:
    tau_sens = [0, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6]
    tau_nosens = [0, 1, 2, 3, 4, 5, 6]

    def __init__(self, subsample_size_test=None, subsample_size_train=None):
        self.dataset, self.models_other_than_rules = exp.base_exp(return_vars=True)
        self.res_dir = 'results/{}'.format(self.dataset)
        out.create_dir(self.res_dir)
        self.res_dir = self.res_dir if not exp.FAIRNESS_CONSTRAINTS else '{}/FC'.format(self.res_dir)
        out.create_dir(self.res_dir)
        self.res_file_path = self.res_dir + '/res_lti.txt'
        self.wiki_parent_path = "Actionable-Explanations/Simple-Explanations-{}".format(self.dataset)
        self.subsample_size_test = subsample_size_test
        self.subsample_size_train = subsample_size_train
        self.sens_group_desc = exp.dataset_info[self.dataset]['sens_f']
        self.prediction_task = exp.dataset_info[self.dataset]['prediction_task']
        self.cost_groups = {cf.ONE_GROUP_IND: "all", 0: self.sens_group_desc[0], 1: self.sens_group_desc[1]}

    def initialize_variables(self, model):
        self.feature_info = model.feature_info
        self.x_control = model.x_control
        self.x_test_original, self.x_train_original = model.x_test, model.x_train
        self.y_test, self.y_train = (model.y_test).astype(bool if self.prediction_task == exp.CLASSIFICATION else float), (model.y_train).astype(bool if self.prediction_task == exp.CLASSIFICATION else float)
        scaler = MinMaxScaler()
        scaler.fit(self.x_train_original)
        self.x_train = scaler.transform(self.x_train_original)
        self.x_test = scaler.transform(self.x_test_original)

        self.sens_group = ~model.x_control[self.sens_group_desc[-1]]
        self.sens_group_train = ~model.x_control_train[self.sens_group_desc[-1]]
        self.sens_group_test = ~model.x_control_test[self.sens_group_desc[-1]]
        
        ##Use these for the remaining analysis
        self.analysis = [("group-efforts", self.sens_group_train)]
        self.cost_funcs, self.feature_val_costs = exp.dataset_info[self.dataset]['cost_funcs'](self.feature_info, 
            self.x_train, self.sens_group_train, exp.dataset_info[self.dataset]['variable_constraints'])
        self.cost_funcs_rev, self.feature_val_costs_rev = exp.dataset_info[self.dataset]['cost_funcs'](self.feature_info, 
            self.x_train, self.sens_group_train, exp.dataset_info[self.dataset]['variable_constraints_rev'])

        self.disparity_table_heading = ["model"]
        self.disparity_table_formats = [None]
        self.disparity_table_values = []

    def get_explanations_test(self, i, index_in_users, optimizer, user, y_train_flipped_pred, filtered_users_gt_labels, filtered_users_pred_labels, 
        filtered_users_sens_group, x_train_flipped, y_train_flipped):
        ### Function to paralellize generation of explanations
        new_feature_vector, utility, effort, false_positive, role_model_gt, anchor = optimizer.sampling_based_explanations(
            user,
            x_train_flipped,
            y_train_flipped,
            y_train_flipped_pred,
            filtered_users_gt_labels[i],
            filtered_users_pred_labels[i],
            user_sens_group=filtered_users_sens_group[i])
        # queue.put((i, index_in_users, user, new_feature_vector, utility, effort, false_positive, role_model_gt))
        return i, index_in_users, user, new_feature_vector, utility, effort, false_positive, role_model_gt, anchor

    def generate_explanations_for_test(self, model, x_train_flipped, y_train_flipped, x_test_flipped, y_test_flipped, threshold_sens, threshold_nosens, cost_funcs, cost_funcs_rev):
        y_test_pred = model.predict(x_test_flipped).astype(bool if self.prediction_task == exp.CLASSIFICATION else float)
        y_train_pred = model.predict(x_train_flipped).astype(bool if self.prediction_task == exp.CLASSIFICATION else float)
        explanations_mask = np.zeros(len(x_test_flipped), dtype=bool)
        explanations_mask[np.where(y_test_pred < self.y_test)[0][:self.subsample_size_test] if self.subsample_size_test is not None else np.where(y_test_pred < self.y_test)[0]] = 1
        filtered_users = x_test_flipped[explanations_mask]
        filtered_users_sens_group = self.sens_group_test[explanations_mask]
        filtered_users_gt_labels = y_test_flipped[explanations_mask]
        filtered_users_pred_labels = y_test_pred[explanations_mask]
        user_utilities = np.zeros(len(filtered_users))
        
        procs, queue = [], Queue()

        x_test_flipped_twice, y_test_flipped_twice = x_test_flipped.copy(), y_test_flipped.copy()

        for i, user in enumerate(filtered_users):
            # if i == 1986:
            #     continue
            user = np.array([user])
            index_in_users = np.where(explanations_mask)[0][i]
            # print("Computing for user", index_in_users)

            optimizer = ge.SamplingMethod(np.array([filtered_users_sens_group[i]]), self.feature_info, cost_funcs, 
                cost_funcs_rev, exp.dataset_info[self.dataset]['variable_constraints'], model, self.dataset)
            # p = Process(target=self.get_explanations_test, args=(queue, i, index_in_users, optimizer, user, y_train_pred, filtered_users_gt_labels, 
            #     filtered_users_pred_labels, filtered_users_sens_group, x_train_flipped, y_train_flipped, ))
            # p.start()
            # procs.append(p)
            # if len(procs) == 5:
                # for p in procs:
            # i, index_in_users, user, new_feature_vector, utility, effort, false_positive, role_model_gt = queue.get()
            i, index_in_users, user, new_feature_vector, utility, effort, false_positive, role_model_gt, anchor = \
                self.get_explanations_test(i, index_in_users, optimizer, user, y_train_pred, filtered_users_gt_labels, 
                    filtered_users_pred_labels, filtered_users_sens_group, x_train_flipped, y_train_flipped)
            print ("User {} retrieved".format(i))
            user_utilities[i] = utility
            x_test_flipped_twice[index_in_users] = new_feature_vector
            y_test_flipped_twice[index_in_users] = role_model_gt


        user_utilities = np.array(user_utilities)
        all_utilities = np.zeros(len(self.x_test))
        all_utilities[explanations_mask] = user_utilities
        out.create_dir('./flipped_datasets')
        out.create_dir('./flipped_datasets/{}'.format(self.dataset))
        np.savetxt('./flipped_datasets/{}/{}_x_test_twice_{}_{}.txt'.format(self.dataset, model.filename(), threshold_sens, threshold_nosens), x_test_flipped_twice)
        np.savetxt('./flipped_datasets/{}/{}_y_test_twice_{}_{}.txt'.format(self.dataset, model.filename(), threshold_sens, threshold_nosens), y_test_flipped_twice)
        np.savetxt('./flipped_datasets/{}/{}_new_utilities_test_{}_{}.txt'.format(self.dataset, model.filename(), threshold_sens, threshold_nosens), all_utilities)

        # summary_of_useful_explanations = "Total explanations given for test set: {} ({} sens group, {} non-sens group)\nUseful explanations: {} ({} sens group, {} non-sens group)".format(
        #     len(filtered_users), len(filtered_users[filtered_users_sens_group]), len(filtered_users[~filtered_users_sens_group]),
        #     np.count_nonzero(user_utilities > 0), np.count_nonzero(user_utilities[filtered_users_sens_group] > 0), np.count_nonzero(user_utilities[~filtered_users_sens_group] > 0))

        # with open(self.res_file_path, 'a') as res_file:
        #     res_file.write("{}\n\n".format(summary_of_useful_explanations))

        # if model.filename() == lm.LogReg().filename():
        #     headings, formats, values = eval_formula.get_disparity_measures(y_test_flipped, y_test_pred, self.sens_group_test, 
        #             np.mean(user_utilities[filtered_users_sens_group]), 
        #             np.mean(user_utilities[~filtered_users_sens_group]), 
        #             return_heading_and_formats=True)
        #     self.disparity_table_heading += headings
        #     self.disparity_table_formats += formats
        # else:
        #     values = eval_formula.get_disparity_measures(y_test_flipped, y_test_pred, self.sens_group_test, 
        #         np.mean(user_utilities[filtered_users_sens_group]), 
        #         np.mean(user_utilities[~filtered_users_sens_group]), 
        #         return_heading_and_formats=False)
        # self.disparity_table_values.append([str(model)] + values)

    def get_flipped_dataset(self, model, string, prediction_task):
        try:
            x = np.loadtxt("./flipped_datasets/{}/{}_all_x_{}.txt".format(self.dataset, model.filename(), string)).astype(float)
            y = np.loadtxt("./flipped_datasets/{}/{}_all_y_{}.txt".format(self.dataset, model.filename(), string)).astype(bool if self.prediction_task == exp.CLASSIFICATION else float)
            return x, y
        except:
            raise ValueError("First run experiment.py to generate flipped {} set".format(string))

    def get_utilities(self, model, string):
        try:
            utilities = np.loadtxt("./flipped_datasets/{}/{}_utilities_{}.txt".format(self.dataset, model.filename(), string)).astype(float)
            return utilities
        except:
            raise ValueError("First run experiment.py to generate {}_utilities_{}.txt".format(model.filename(), string))

    def run(self):
        learning_env = dec_rule_env.DecRuleEnv(self.dataset, self.sens_group_desc)
        learning_env.load_data(feature_engineering=True)
        self.initialize_variables(learning_env)

        models_to_train = self.models_other_than_rules

        for model in models_to_train:
            model_start = time.time()
            exists, loaded_clf = aeio.load_model(model, self.dataset)
            if exists:
                model = loaded_clf
                print ("Loaded {}...".format(str(model)))
            else:
                print ("Training {}...".format(str(model)))
                model.train(self.x_train, self.y_train)
                aeio.persist_model(model, self.dataset)
            model_end = time.time()
            x_test_flipped, y_test_flipped = self.get_flipped_dataset(model, 'test', self.prediction_task)
            x_test_utilities = self.get_utilities(model, 'test')
            # y_test_flipped_preds = model.predict(x_test_flipped).astype(bool)

            x_train_flipped, y_train_flipped = self.get_flipped_dataset(model, 'train', self.prediction_task)
            x_train_utilities = self.get_utilities(model, 'train')
            for threshold_sens in LongTermImpact.tau_sens:
                # get new x_test and x_train
                for threshold_nosens in LongTermImpact.tau_nosens:
                    new_x_train, new_y_train, new_x_test, new_y_test = self.x_train.copy(), self.y_train.copy(), self.x_test.copy(), self.y_test.copy()
                    indices_to_flip_train_sens, indices_to_flip_test_sens = (np.where(np.logical_and(x_train_utilities > threshold_sens, self.sens_group_train))[0], 
                        np.where(np.logical_and(x_test_utilities > threshold_sens, self.sens_group_test))[0])
                    indices_to_flip_train_nosens, indices_to_flip_test_nosens = (np.where(np.logical_and(x_train_utilities > threshold_nosens, ~self.sens_group_train))[0], 
                        np.where(np.logical_and(x_test_utilities > threshold_nosens, ~self.sens_group_test))[0])
                    new_x_test[indices_to_flip_test_sens,:], new_y_test[indices_to_flip_test_sens] = x_test_flipped[indices_to_flip_test_sens,:], y_test_flipped[indices_to_flip_test_sens]
                    new_x_test[indices_to_flip_test_nosens,:], new_y_test[indices_to_flip_test_nosens] = x_test_flipped[indices_to_flip_test_nosens,:], y_test_flipped[indices_to_flip_test_nosens]
                    new_x_train[indices_to_flip_train_sens,:], new_y_train[indices_to_flip_train_sens] = x_train_flipped[indices_to_flip_train_sens,:], y_train_flipped[indices_to_flip_train_sens]
                    new_x_train[indices_to_flip_train_nosens,:], new_y_train[indices_to_flip_train_nosens] = x_train_flipped[indices_to_flip_train_nosens,:], y_train_flipped[indices_to_flip_train_nosens]
                    # generate explanations for this new x_test by picking role models from new_x_train
                    cost_funcs, _ = exp.dataset_info[self.dataset]['cost_funcs'](self.feature_info, 
                        new_x_train, self.sens_group_train, exp.dataset_info[self.dataset]['variable_constraints'])
                    cost_funcs_rev, _ = exp.dataset_info[self.dataset]['cost_funcs'](self.feature_info, 
                        new_x_train, self.sens_group_train, exp.dataset_info[self.dataset]['variable_constraints_rev'])
                    self.generate_explanations_for_test(model, new_x_train, new_y_train, new_x_test, new_y_test, threshold_sens, threshold_nosens, cost_funcs, cost_funcs_rev)


if __name__ == "__main__":
    start = time.time()
    lti = LongTermImpact(subsample_size_test=None, subsample_size_train=None)
    lti.run()
    end = time.time()
    print ("Time taken: {:.3f}".format(end - start))
