import sys
import time
from sys import maxsize
from itertools import product, repeat
import numpy as np
from sklearn.metrics import pairwise_distances

import act_exp_io as aeio
import cost_funcs as cf

sys.path.insert(0, '../util/')
import output as out
from datasets import file_util as fu

USE_VAR_WEIGHT = 1.0
INF = 1000000000000

def get_conditions(feature_infos, var_indices, var_values):
    assignments = {var_ind: var_val for var_ind, var_val in zip(var_indices, var_values)}
    # print ("\n\n{}\n".format(assignments))
    conditions = {}

    var_offset = 0
    for fname, ftype, flabels in feature_infos:
        feature_assignments = [(vind, assignments[vind + var_offset]) for vind in range(len(flabels)) if var_offset + vind in assignments]
        if len(feature_assignments) == 0:
            var_offset += len(flabels)
            continue

        if ftype == fu.ATTR_CONT or ftype == fu.ATTR_CAT_BIN:
            assert len(feature_assignments) == 1
            conditions[fname] = (var_offset, feature_assignments[0][1])
            var_offset += len(flabels)
            continue

        pos_assignments = [vind for vind, val in feature_assignments if val == 1]
        if len(pos_assignments) == 0:
            # TODO: try to avoid setting false indicators no true indicator is set
            # TODO: check that the recommendation corresponds to the original feature value
            var_offset += len(flabels)
            continue

        assert len(pos_assignments) == 1 or ftype == fu.ATTR_CAT_BIN

        if ftype == fu.ATTR_CAT_BIN:
            is_pos = len(pos_assignments) == 1
            target_index, target_val = cf.get_bin_index_val(flabels, is_pos)
        else:
            target_index = pos_assignments[0]
            _, target_val = flabels[target_index]
        conditions[fname] = (target_index, target_val)

        var_offset += len(flabels)
    # print ("\n\nConditions: {}\n\n".format(conditions))
    return conditions


class SamplingMethod:

    def __init__(self, cost_sens_group, feature_info, cost_funcs, cost_funcs_rev, 
        variable_constraints, model, dataset):
        self.cost_funcs = cost_funcs
        self.cost_funcs_rev = cost_funcs_rev
        self.cost_sens_group = cost_sens_group
        self.feature_info = feature_info
        self.variable_constraints = variable_constraints
        self.model = model
        self.dataset = dataset

    def effort_measure(self, row1, row2):
        var_indices = list(np.where(row1 != row2)[0])
        var_vals = list(row2[var_indices])
        explanation = get_conditions(self.feature_info, var_indices, var_vals)
        # print ("\nexplanation: {}\n".format(explanation))
        user_effort = cf.compute_efforts(self.feature_info, self.cost_funcs, self.cost_funcs_rev, [row1], explanation, self.cost_sens_group)[0]
        return user_effort

    def get_possible_role_models(self, user, user_ground_truth, user_predicted_val, users, users_ground_truth, users_predicted_val):
        print (users.shape)
        ## Keeping track of focal points/anchors
        indices_of_users = np.array(list(zip(*enumerate(users)))[0])
        # Forcing role models to have Higher ground truth label than that of user
        mask_higher_gt = np.where(users_predicted_val >= user_predicted_val)[0]
        users = users[mask_higher_gt]
        users_predicted_val = users_predicted_val[mask_higher_gt]
        users_ground_truth = users_ground_truth[mask_higher_gt]
        indices_of_users = indices_of_users[mask_higher_gt]
        assert np.all(users_predicted_val >= user_predicted_val)

        # #### TODO: dirty, make it nicer
        if self.dataset != 'CrimesCommunities':
            immutable_cols = cf.get_immutable_cols(self.variable_constraints, self.feature_info)
            if len(immutable_cols) > 0:
                assert users[:,immutable_cols].shape[1] == user[immutable_cols].shape[0] # immutable features should be same for user and population
                mask_immutable_cols_same = np.where(np.all(users[:,immutable_cols] == user[immutable_cols], axis=1))[0]
                users = users[mask_immutable_cols_same] # user indices where values are same and hence can be legit role models
                users_ground_truth = users_ground_truth[mask_immutable_cols_same]
                users_predicted_val = users_predicted_val[mask_immutable_cols_same]
                indices_of_users = indices_of_users[mask_immutable_cols_same]
        print (users.shape)

        dir_up_cols_generator = cf.get_up_cols(self.variable_constraints, self.feature_info)
        for dir_up_cols in dir_up_cols_generator:
            print (dir_up_cols)
            assert users[:,dir_up_cols].shape[1] == user[dir_up_cols].shape[0] # features should be same for user and population
            ######### CAUTION: Dirty implementation below; please forgive me
            ### Only for binary features
            # xor = np.logical_xor(users[:,dir_up_cols], user[dir_up_cols])
            # true_row_idx, true_col_idx = np.where(xor) # xor between one hot encoded vectors; since a user's value cannot go down, the idx of 1 in one-hot encoded vector should be greater than or equal to all indices of legit role models
            # rows_to_exclude = true_row_idx[np.where(true_col_idx < np.where(user[dir_up_cols])[0])[0]]
            ### Only for cont features
            rows_to_exclude = np.where(np.all(users[:,dir_up_cols] < user[dir_up_cols], axis=1))[0] # delete rows where users have value lesser than the current user's value

            users = np.delete(users, rows_to_exclude, axis=0)
            users_ground_truth = np.delete(users_ground_truth, rows_to_exclude, axis=0)
            users_predicted_val = np.delete(users_predicted_val, rows_to_exclude, axis=0)
            indices_of_users = np.delete(indices_of_users, rows_to_exclude)
        # print (users.shape)
        return users, users_ground_truth, users_predicted_val, indices_of_users

    def is_in_users(self, x):
        presence_of_x = np.all(self.all_users == x, axis=1)
        if np.count_nonzero(presence_of_x) == 0:
            return False, None
        idx = np.where(presence_of_x)[0][0]
        return True, self.all_users_gt[idx]

    def get_explanations(self, user, user_ground_truth, user_predicted_val, role_model, role_model_gt, role_model_pred, user_sens_group):
        w = np.ones(len(user))
        w[np.where(user == role_model)[0]] = 0
        immutable_cols = cf.get_immutable_cols(self.variable_constraints, self.feature_info)
        w[immutable_cols] = 0
        print ("indices where unequal: {}, w at those indices: {}".format(np.where(user != role_model)[0], w[np.where(user != role_model)[0]]))
        print ("indices where immutable: {}, w at those indices: {}".format(immutable_cols, w[immutable_cols]))
        w_original = w.copy()
        x_new = w * role_model + (1 - w) * user
        y_new = role_model_gt
        while True:
            x_new_pred = self.model.predict([x_new])[0]
            is_a_user, corresponding_gt = self.is_in_users(x_new)
            x_new_gt = corresponding_gt if is_a_user else x_new_pred
            u = cf.compute_utility(x_new_gt, x_new_pred, user_ground_truth, user_predicted_val, user_sens_group, self.effort_measure(user, x_new))
            if np.all(x_new == role_model):
                assert is_a_user # sanity checks
                assert corresponding_gt == role_model_gt # sanity checks
            u_ks = []
            w_ks = []
            idx = 0
            for fname, _, fvals in self.feature_info:
                if len(np.where(w[list(range(idx, idx+len(fvals)))] == 1)[0]) > 0: # if any of the boolean one hot encoded labels is 1 then w for that variable is 1
                    assert np.all(w[list(range(idx, idx+len(fvals)))]) if len(fvals) <= 2 else len(np.where(w[list(range(idx, idx+len(fvals)))] == 1)[0]) == 2 # the user differes from role model in atleast 2 positions if it's a one hot encoded vector
                    w_dash = w.copy()
                    w_dash[list(range(idx, idx+len(fvals)))] = 0
                    w_ks.append(w_dash)
                    x_dash = w_dash * role_model + (1 - w_dash) * user
                    assert np.all(x_dash[immutable_cols] == user[immutable_cols]) # sanity check
                    x_dash_pred = self.model.predict([x_dash])[0]
                    is_a_user, corresponding_gt = self.is_in_users(x_dash)
                    x_dash_true_label = corresponding_gt if is_a_user else x_dash_pred
                    u_k = cf.compute_utility(x_dash_true_label, x_dash_pred, user_ground_truth, 
                            user_predicted_val, user_sens_group, self.effort_measure(user, x_dash))
                    u_ks.append(u_k)

                    if x_dash_true_label == x_new_gt and x_new_pred == x_dash_pred:
                        # if no change in reward function then dropping a feature should only increase utility
                        assert u_k >= u
                else:
                    u_ks.append(-INF)
                    w_ks.append(w)
                idx += len(fvals)
            j = np.argmax(u_ks)
            u_j = u_ks[j]
            w_j = w_ks[j]
            print ("u_j: {}, u: {}".format(u_j,u))
            if u_j >= u:
                w = w_j
                print ("indices where immutable: {}, w at those indices: {}".format(immutable_cols, w[immutable_cols]))
                x_new = w * role_model + (1 - w) * user
                assert np.all(x_new[immutable_cols] == user[immutable_cols]) # sanity check
            if u > u_j:
                break
            # time.sleep(2)
        print (np.all(w == w_original))
        x_new_effort = self.effort_measure(user, x_new)
        x_new_pred = self.model.predict([x_new])[0]
        is_a_user, corresponding_gt = self.is_in_users(x_new)
        x_new_utility = cf.compute_utility(corresponding_gt if is_a_user else x_new_pred, x_new_pred, user_ground_truth, user_predicted_val, user_sens_group, x_new_effort)
        
        dir_up_cols_generator = cf.get_up_cols(self.variable_constraints, self.feature_info)
        for dir_up_cols in dir_up_cols_generator:
            assert np.all(x_new[dir_up_cols] >= user[dir_up_cols]) # education
        assert np.all(x_new[immutable_cols] == user[immutable_cols])
        return x_new, x_new_utility, x_new_effort, corresponding_gt if is_a_user else x_new_pred, x_new_pred

    def sampling_based_explanations(self, user, users, users_ground_truth, users_predicted_val, user_ground_truth, user_predicted_val, user_sens_group, return_only_user=False):
        self.all_users, self.all_users_gt = users, users_ground_truth
        right_users, corresponding_gt, corresponding_preds, indices_of_users = self.get_possible_role_models(user.flatten(), user_ground_truth, user_predicted_val, 
            users, users_ground_truth, users_predicted_val)
        if right_users.shape[0] == 0:
            return user[0], 0, 0, user_ground_truth == False and user_predicted_val == True, user_ground_truth, np.nan
        efforts = pairwise_distances(user, right_users, metric=self.effort_measure, n_jobs=-1).flatten()
        utilities = cf.compute_utility(corresponding_gt, corresponding_preds, np.array([user_ground_truth] * len(right_users)), 
            np.array([user_predicted_val] * len(right_users)), np.array([user_sens_group] * len(right_users)), efforts)
        idx = np.argmax(utilities)
        role_model = right_users[idx]
        role_model_gt = corresponding_gt[idx]
        role_model_pred = corresponding_preds[idx]
        role_model_utility = utilities[idx]
        role_model_effort = efforts[idx]
        anchor = indices_of_users[idx]
        if return_only_user:
            # used for effort_reward_function_plots.py
            return (role_model, role_model_gt, role_model_pred)
        # x_new, x_new_utility, x_new_effort, x_new_gt, x_new_pred = self.get_explanations(user.flatten(), user_ground_truth, user_predicted_val, 
        #     role_model.flatten(), role_model_gt, role_model_pred, user_sens_group)
        return (role_model, role_model_utility, role_model_effort, role_model_gt == False and role_model_pred == True, role_model_gt, anchor)