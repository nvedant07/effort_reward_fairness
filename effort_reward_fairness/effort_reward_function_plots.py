import long_term_impact as lti
import act_exp_io as aeio
import experiment as exp
import group_explanations as ge
from learning_env import dec_rule_env
import cost_funcs as cf

import time, sys, os
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals import joblib

sys.path.insert(0, "../util/")
import output as out


class EffortRewardPlots(lti.LongTermImpact):

    def __init__(self, subsample_size_test=None, subsample_size_train=100):
        super(EffortRewardPlots, self).__init__(subsample_size_test, subsample_size_train)
        self.res_file_path = self.res_dir + '/res_function_plots.txt'
        self.effort_deltas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        self.reward_deltas = [1, 3, 5, 7, 9, 11, 13]

    def set_vars(self, test_or_train):
        self.role_model_users = self.x_train # This should not change
        self.role_model_users_gt_labels = self.y_train # This should not change
        self.role_model_users_sens_group = self.sens_group_train # This should not change
        self.users = self.x_test if test_or_train == 'test' else self.x_train # change this based on which group's explanations are needed (test or train)
        self.users_sens_group = self.sens_group_test if test_or_train == 'test' else self.sens_group_train # change this based on which group's explanations are needed (test or train)
        self.users_gt = self.y_test if test_or_train == 'test' else self.y_train # change this based on which group's explanations are needed (test or train)
        self.subsample_size = self.subsample_size_test if test_or_train == 'test' else self.subsample_size_train
        self.variable_constraints = exp.dataset_info[self.dataset]['variable_constraints']

    def is_in_users(self, x):
        presence_of_x = np.all(self.users == x, axis=1)
        if np.count_nonzero(presence_of_x) == 0:
            return False, None
        idx = np.where(presence_of_x)[0][0]
        return True, self.users_gt[idx]

    def effort_measure(self, row1, row2, cost_sens_group):
        var_indices = list(np.where(row1 != row2)[0])
        var_vals = list(row2[var_indices])
        explanation = ge.get_conditions(self.feature_info, var_indices, var_vals)
        # print ("\nexplanation: {}\n".format(explanation))
        user_effort = cf.compute_efforts(self.feature_info, self.cost_funcs, self.cost_funcs_rev, [row1], explanation, cost_sens_group)[0]
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

    def sampling_based_explanations(self, user, users, users_ground_truth, users_predicted_val, user_ground_truth, 
        user_predicted_val, user_sens_group, cost_sens_group, variable_to_optimize, variable_to_threshold, threshold_value):
        right_users, corresponding_gt, corresponding_preds, indices_of_users = self.get_possible_role_models(user.flatten(), user_ground_truth, user_predicted_val, 
            users, users_ground_truth, users_predicted_val)
        idx_to_remove = np.where(np.all(right_users == user.flatten(), axis=1))[0] # remove the user from this list
        right_users, corresponding_gt, corresponding_preds, indices_of_users = np.delete(right_users, idx_to_remove, axis=0), np.delete(corresponding_gt, idx_to_remove), \
            np.delete(corresponding_preds, idx_to_remove), np.delete(indices_of_users, idx_to_remove)
        if right_users.shape[0] == 0:
            return user[0], 0, 0, 0

        efforts = pairwise_distances(X=user, Y=right_users, metric=self.effort_measure, n_jobs=-1, **{'cost_sens_group': cost_sens_group}).flatten()
        utilities = cf.compute_utility(corresponding_gt, corresponding_preds, np.array([user_ground_truth] * len(right_users)), 
            np.array([user_predicted_val] * len(right_users)), np.array([user_sens_group] * len(right_users)), efforts)
        rewards = cf.compute_reward(corresponding_gt, corresponding_preds, np.array([user_ground_truth] * len(right_users)), 
            np.array([user_predicted_val] * len(right_users)), np.array([user_sens_group] * len(right_users)))

        if variable_to_threshold == 'effort':
            mask = efforts <= threshold_value
        elif variable_to_threshold == 'reward':
            mask = rewards >= threshold_value
        else:
            raise ValueError("Not the right variable to threshold")
        if np.count_nonzero(mask) > 0:
            # print ("Before: {}, {}, {}\n".format(rewards, efforts, utilities))
            efforts, utilities, rewards, right_users = efforts[mask], utilities[mask], rewards[mask], right_users[mask]
            # print ("After: {}, {}, {}\n".format(rewards, efforts, utilities))
            # print ("Threshold on {} = {}\n".format(variable_to_threshold, threshold_value))

            if variable_to_optimize == 'effort':
                idx = np.argmin(efforts)
            elif variable_to_optimize == 'reward':
                idx = np.argmax(rewards)
            elif variable_to_optimize == 'utility':
                idx = np.argmax(utilities)
            # print ('Effort: {}, Reward: {}, Utility: {}'.format(efforts[idx], rewards[idx], utilities[idx]))
            return right_users[idx], efforts[idx], rewards[idx], utilities[idx]
        else:
            if variable_to_optimize == 'reward' or variable_to_optimize == 'utility':
                # print ("Oh no!")
                return user.flatten(), 0, 0, 0
            elif variable_to_optimize == 'effort':
                idx = np.argmax(efforts)
                # print ('Effort: {}'.format(efforts[idx]))
                return right_users[idx], efforts[idx], rewards[idx], utilities[idx]

    def generate_new_feature_vector(self, model, user, user_ground_truth, user_predicted_val, role_model, role_model_gt, role_model_pred, 
        user_sens_group, optimizer, variable_to_optimize, variable_to_threshold, threshold_value):
        w = np.ones(len(user))
        w[np.where(user == role_model)[0]] = 0
        immutable_cols = cf.get_immutable_cols(self.variable_constraints, self.feature_info)
        w[immutable_cols] = 0
        w_original = w.copy()
        x_new = w * role_model + (1 - w) * user
        while True:
            x_new_pred = model.predict([x_new])[0]
            is_a_user, corresponding_gt = self.is_in_users(x_new)
            x_new_gt = corresponding_gt if is_a_user else x_new_pred

            u = cf.compute_utility(x_new_gt, x_new_pred, user_ground_truth, user_predicted_val, user_sens_group, self.effort_measure(user, x_new, optimizer.cost_sens_group))
            r = cf.compute_reward(x_new_gt, x_new_pred, user_ground_truth, user_predicted_val, user_sens_group)
            e = self.effort_measure(user, x_new, optimizer.cost_sens_group)
            # print ('effort: {}, reward: {}, utility: {}'.format(e, r, u))

            if np.all(x_new == role_model):
                assert is_a_user # sanity checks
                assert corresponding_gt == role_model_gt # sanity checks
            u_ks, r_ks, e_ks = [], [], []
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
                    x_dash_pred = model.predict([x_dash])[0]
                    is_a_user, corresponding_gt = self.is_in_users(x_dash)
                    x_dash_true_label = corresponding_gt if is_a_user else x_dash_pred
                    u_k = cf.compute_utility(x_dash_true_label, x_dash_pred, user_ground_truth, 
                            user_predicted_val, user_sens_group, self.effort_measure(user, x_dash, optimizer.cost_sens_group))
                    r_k = cf.compute_reward(x_dash_true_label, x_dash_pred, user_ground_truth, user_predicted_val, user_sens_group)
                    e_k = self.effort_measure(user, x_dash, optimizer.cost_sens_group)
                    u_ks.append(u_k)
                    r_ks.append(r_k)
                    e_ks.append(e_k)

                    if x_dash_true_label == x_new_gt and x_new_pred == x_dash_pred:
                        # if no change in reward function then dropping a feature should only increase utility
                        assert u_k >= u
                else:
                    assert np.all(w[list(range(idx, idx+len(fvals)))] == 0)
                    u_ks.append(-ge.INF)
                    r_ks.append(-ge.INF)
                    e_ks.append(ge.INF)
                    w_ks.append(w)
                idx += len(fvals)

            u_ks, r_ks, e_ks, w_ks = np.array(u_ks), np.array(r_ks), np.array(e_ks), np.array(w_ks)
            if variable_to_threshold == 'effort':
                mask = e_ks <= threshold_value
            elif variable_to_threshold == 'reward':
                mask = r_ks >= threshold_value
            else:
                raise ValueError("Not the right variable to threshold")

            if np.count_nonzero(mask) > 0:
                u_ks, r_ks, e_ks, w_ks = u_ks[mask], r_ks[mask], e_ks[mask], w_ks[mask]
            
            if variable_to_optimize == 'effort':
                if np.all(e_ks == ge.INF):
                    return x_new, u, e, r, x_new_gt, x_new_pred
                j = np.argmin(e_ks)
                u_j, e_j, r_j = u_ks[j], e_ks[j], r_ks[j]
                w_j = w_ks[j]
                print ("e_j: {}, e: {}".format(e_j,e))
                if e_j <= e or (e > threshold_value if variable_to_threshold == 'effort' else r > threshold_value):
                    w = w_j
                    x_new = w * role_model + (1 - w) * user
                    assert np.all(x_new[immutable_cols] == user[immutable_cols]) # sanity check
                elif e < e_j:
                    break
            elif variable_to_optimize == 'reward':
                j = np.argmax(r_ks)
                u_j, e_j, r_j = u_ks[j], e_ks[j], r_ks[j]
                w_j = w_ks[j]
                print ("r_j: {}, r: {}".format(r_j,r))
                if r_j >= r or (e > threshold_value if variable_to_threshold == 'effort' else r > threshold_value):
                    w = w_j
                    x_new = w * role_model + (1 - w) * user
                    assert np.all(x_new[immutable_cols] == user[immutable_cols]) # sanity check
                elif r > r_j:
                    break
            elif variable_to_optimize == 'utility':
                j = np.argmax(u_ks)
                u_j = u_ks[j]
                w_j = w_ks[j]
                print ("u_j: {}, u: {}".format(u_j,u))
                if u_j >= u or (e > threshold_value if variable_to_threshold == 'effort' else r > threshold_value):
                    w = w_j
                    x_new = w * role_model + (1 - w) * user
                    assert np.all(x_new[immutable_cols] == user[immutable_cols]) # sanity check
                elif u > u_j:
                    break
        print (np.all(w == w_original))
        x_new_effort = self.effort_measure(user, x_new, optimizer.cost_sens_group)
        x_new_pred = model.predict([x_new])[0]
        is_a_user, corresponding_gt = self.is_in_users(x_new)
        x_new_gt = corresponding_gt if is_a_user else x_new_pred
        x_new_reward = cf.compute_reward(x_new_gt, x_new_pred, user_ground_truth, user_predicted_val, user_sens_group)
        x_new_utility = cf.compute_utility(x_new_gt, x_new_pred, user_ground_truth, user_predicted_val, user_sens_group, x_new_effort)
        
        dir_up_cols_generator = cf.get_up_cols(self.variable_constraints, self.feature_info)
        for dir_up_cols in dir_up_cols_generator:
            assert np.all(x_new[dir_up_cols] >= user[dir_up_cols]) # education
        assert np.all(x_new[immutable_cols] == user[immutable_cols])
        if variable_to_threshold == 'effort':
            assert x_new_effort <= threshold_value
        elif variable_to_threshold == 'reward':
            assert x_new_reward <= threshold_value
        return x_new, x_new_utility, x_new_effort, x_new_reward, x_new_gt, x_new_pred

    def run(self, test_or_train):
        learning_env = dec_rule_env.DecRuleEnv(self.dataset, self.sens_group_desc)
        learning_env.load_data(feature_engineering=True)
        self.initialize_variables(learning_env)
        self.set_vars(test_or_train)

        with open(self.res_file_path, 'w') as res_file:
            res_file.write("= Effort, Reward and Utilities as functions of one another =\n\n")

        model_to_utility_sens, model_to_utility_nosens = {}, {}
        model_to_reward_sens, model_to_reward_nosens = {}, {}
        model_to_effort_sens, model_to_effort_nosens = {}, {}

        for model in self.models_other_than_rules:
            sens_utils_with_effort, nosens_utils_with_effort = [], []
            sens_reward_with_effort, nosens_reward_with_effort = [], []
            sens_effort_with_reward, nosens_effort_with_reward = [], []
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

            y_test_pred = model.predict(self.users).astype(bool if exp.dataset_info[self.dataset]['prediction_task'] == exp.CLASSIFICATION else float)
            y_train_pred = model.predict(self.role_model_users).astype(bool if exp.dataset_info[self.dataset]['prediction_task'] == exp.CLASSIFICATION else float)
            
            print ("Model: {}, MAE: {}, MSE: {}".format(model, mean_absolute_error(self.users_gt, y_test_pred), mean_squared_error(self.users_gt, y_test_pred)))
            continue

            self.role_model_users_pred = y_train_pred # This should not change
            self.users_preds = y_test_pred if test_or_train == 'test' else y_train_pred # change this based on which group's explanations are needed (test or train)

            sub_filter_sens = np.zeros(len(self.users), dtype=bool)
            sub_filter_nosens = np.zeros(len(self.users), dtype=bool)
            sub_filter_sens[np.where(np.logical_and(self.users_sens_group, self.users_preds < self.users_gt))[0][:self.subsample_size] if \
                self.subsample_size is not None else np.where(np.logical_and(self.users_sens_group, self.users_preds < self.users_gt))[0]] = 1
            sub_filter_nosens[np.where(np.logical_and(~self.users_sens_group, self.users_preds < self.users_gt))[0][:self.subsample_size] if \
                self.subsample_size is not None else np.where(np.logical_and(~self.users_sens_group, self.users_preds < self.users_gt))[0]] = 1
            filtered_users_sens = self.users[sub_filter_sens]
            user_gt_labels_sens = self.users_gt[sub_filter_sens]
            user_predicted_labels_sens = self.users_preds[sub_filter_sens]
            filtered_users_nosens = self.users[sub_filter_nosens]
            user_gt_labels_nosens = self.users_gt[sub_filter_nosens]
            user_predicted_labels_nosens = self.users_preds[sub_filter_nosens]

            ind_start = time.time()

            for delta in self.effort_deltas:
                sens_rewards, sens_utils, nosens_rewards, nosens_utils = [], [], [], []
                for i, user in enumerate(filtered_users_sens):
                    print("Computing for user", np.where(sub_filter_sens)[0][i])
                    user = np.array([user])
                    index_in_users = np.where(sub_filter_sens)[0][i]

                    # optimizer = ge.SamplingMethod(np.array([1]), self.feature_info, self.cost_funcs, self.cost_funcs_rev, 
                    #     exp.dataset_info[self.dataset]['variable_constraints'], model, self.dataset)
                    # role_model, role_model_gt, role_model_pred = optimizer.sampling_based_explanations(
                    #     user, 
                    #     self.role_model_users, 
                    #     self.role_model_users_gt_labels,
                    #     self.role_model_users_pred,
                    #     user_gt_labels_sens[i],
                    #     user_predicted_labels_sens[i],
                    #     user_sens_group=1,
                    #     return_only_user=True)
                    role_model, role_model_effort, role_model_reward, role_model_utility = \
                        self.sampling_based_explanations(
                            user, 
                            self.role_model_users, 
                            self.role_model_users_gt_labels,
                            self.role_model_users_pred,
                            user_gt_labels_sens[i],
                            user_predicted_labels_sens[i],
                            user_sens_group=1,
                            cost_sens_group=np.array([1]),
                            variable_to_optimize='reward',
                            variable_to_threshold='effort',
                            threshold_value=delta
                        )
                    assert role_model_utility == role_model_reward - role_model_effort
                    sens_rewards.append(role_model_reward)
                    print ("[Sens] Model: {}, Effort threshold: {}, Effort value: {}, Max Reward: {}".format(model, delta,role_model_effort, role_model_reward))
                    break
                    # role_model, role_model_effort, role_model_reward, role_model_utility = \
                    #     self.sampling_based_explanations(
                    #         user, 
                    #         self.role_model_users, 
                    #         self.role_model_users_gt_labels,
                    #         self.role_model_users_pred,
                    #         user_gt_labels_sens[i],
                    #         user_predicted_labels_sens[i],
                    #         user_sens_group=1,
                    #         cost_sens_group=np.array([1]),
                    #         variable_to_optimize='utility',
                    #         variable_to_threshold='effort',
                    #         threshold_value=delta
                    #     )
                    # sens_utils.append(role_model_utility)

                    # x_new, x_new_utility, x_new_effort, x_new_reward, x_new_gt, x_new_pred = \
                    #     self.generate_new_feature_vector(model, user.flatten(), self.users_gt[index_in_users], self.users_preds[index_in_users], 
                    #         role_model, role_model_gt, role_model_pred, 
                    #         1, optimizer, 'reward', 
                    #         'effort', delta)
                    # sens_rewards.append(x_new_reward)
                    # x_new, x_new_utility, x_new_effort, x_new_reward, x_new_gt, x_new_pred = \
                    #     self.generate_new_feature_vector(model, user.flatten(), self.users_gt[index_in_users], self.users_preds[index_in_users], 
                    #         role_model, role_model_gt, role_model_pred, 
                    #         1, optimizer, 'utility', 
                    #         'effort', delta)
                    # sens_utils.append(x_new_utility)
                    
                    dir_up_cols_generator = cf.get_up_cols(exp.dataset_info[self.dataset]['variable_constraints'], self.feature_info)
                    for dir_up_cols in dir_up_cols_generator:
                        assert np.all(role_model[dir_up_cols] >= user.flatten()[dir_up_cols]) # sanity check

                for i, user in enumerate(filtered_users_nosens):
                    print("Computing for user", np.where(sub_filter_nosens)[0][i])
                    user = np.array([user])
                    index_in_users = np.where(sub_filter_nosens)[0][i]

                    # optimizer = ge.SamplingMethod(np.array([0]), self.feature_info, self.cost_funcs, self.cost_funcs_rev, 
                    #     exp.dataset_info[self.dataset]['variable_constraints'], model, self.dataset)
                    # role_model, role_model_gt, role_model_pred = optimizer.sampling_based_explanations(
                    #     user, 
                    #     self.role_model_users, 
                    #     self.role_model_users_gt_labels,
                    #     self.role_model_users_pred,
                    #     user_gt_labels_nosens[i],
                    #     user_predicted_labels_nosens[i],
                    #     user_sens_group=0,
                    #     return_only_user=True)

                    role_model, role_model_effort, role_model_reward, role_model_utility = \
                        self.sampling_based_explanations(
                            user, 
                            self.role_model_users, 
                            self.role_model_users_gt_labels,
                            self.role_model_users_pred,
                            user_gt_labels_nosens[i],
                            user_predicted_labels_nosens[i],
                            user_sens_group=0,
                            cost_sens_group=np.array([0]),
                            variable_to_optimize='reward',
                            variable_to_threshold='effort',
                            threshold_value=delta
                        )
                    assert role_model_utility == role_model_reward - role_model_effort
                    nosens_rewards.append(role_model_reward)
                    print ("[Nosens] Model: {}, Effort threshold: {}, Effort value: {}, Max Reward: {}".format(model, delta, role_model_effort, role_model_reward))
                    break
                    # role_model, role_model_effort, role_model_reward, role_model_utility = \
                    #     self.sampling_based_explanations(
                    #         user, 
                    #         self.role_model_users, 
                    #         self.role_model_users_gt_labels,
                    #         self.role_model_users_pred,
                    #         user_gt_labels_nosens[i],
                    #         user_predicted_labels_nosens[i],
                    #         user_sens_group=0,
                    #         cost_sens_group=np.array([0]),
                    #         variable_to_optimize='utility',
                    #         variable_to_threshold='effort',
                    #         threshold_value=delta
                    #     )
                    # nosens_utils.append(role_model_utility)

                    # x_new, x_new_utility, x_new_effort, x_new_reward, x_new_gt, x_new_pred = \
                    #     self.generate_new_feature_vector(model, user.flatten(), self.users_gt[index_in_users], self.users_preds[index_in_users], 
                    #         role_model, role_model_gt, role_model_pred, 
                    #         0, optimizer, 'reward', 
                    #         'effort', delta)
                    # nosens_rewards.append(x_new_reward)
                    # x_new, x_new_utility, x_new_effort, x_new_reward, x_new_gt, x_new_pred = \
                    #     self.generate_new_feature_vector(model, user.flatten(), self.users_gt[index_in_users], self.users_preds[index_in_users], 
                    #         role_model, role_model_gt, role_model_pred, 
                    #         0, optimizer, 'utility', 
                    #         'effort', delta)
                    # nosens_utils.append(x_new_utility)

                    dir_up_cols_generator = cf.get_up_cols(exp.dataset_info[self.dataset]['variable_constraints'], self.feature_info)
                    for dir_up_cols in dir_up_cols_generator:
                        assert np.all(role_model[dir_up_cols] >= user.flatten()[dir_up_cols]) # sanity check
                # sens_utils_with_effort.append(np.mean(sens_utils))
                sens_reward_with_effort.append(np.mean(sens_rewards))
                # nosens_utils_with_effort.append(np.mean(nosens_utils))
                nosens_reward_with_effort.append(np.mean(nosens_rewards))

            sens_reward_with_effort, nosens_reward_with_effort = np.array(sens_reward_with_effort), np.array(nosens_reward_with_effort)
            sens_utils_with_effort, nosens_utils_with_effort = sens_reward_with_effort - self.effort_deltas, nosens_reward_with_effort - self.effort_deltas
            model_to_utility_sens[model] = [self.effort_deltas, sens_utils_with_effort]
            model_to_utility_nosens[model] = [self.effort_deltas, nosens_utils_with_effort]
            model_to_reward_sens[model] = [self.effort_deltas, sens_reward_with_effort]
            model_to_reward_nosens[model] = [self.effort_deltas, nosens_reward_with_effort]

            for delta in self.reward_deltas:
                sens_efforts, nosens_efforts = [], []
                for i, user in enumerate(filtered_users_sens):
                    print("Computing for user", np.where(sub_filter_sens)[0][i])
                    user = np.array([user])
                    index_in_users = np.where(sub_filter_sens)[0][i]

                    # optimizer = ge.SamplingMethod(np.array([1]), self.feature_info, self.cost_funcs, self.cost_funcs_rev, 
                    #     exp.dataset_info[self.dataset]['variable_constraints'], model, self.dataset)
                    # role_model, role_model_gt, role_model_pred = optimizer.sampling_based_explanations(
                    #     user, 
                    #     self.role_model_users, 
                    #     self.role_model_users_gt_labels,
                    #     self.role_model_users_pred,
                    #     user_gt_labels_sens[i],
                    #     user_predicted_labels_sens[i],
                    #     user_sens_group=1,
                    #     return_only_user=True)

                    role_model, role_model_effort, role_model_reward, role_model_utility = \
                        self.sampling_based_explanations(
                            user, 
                            self.role_model_users, 
                            self.role_model_users_gt_labels,
                            self.role_model_users_pred,
                            user_gt_labels_sens[i],
                            user_predicted_labels_sens[i],
                            user_sens_group=1,
                            cost_sens_group=np.array([1]),
                            variable_to_optimize='effort',
                            variable_to_threshold='reward',
                            threshold_value=delta
                        )
                    sens_efforts.append(role_model_effort)

                    # x_new, x_new_utility, x_new_effort, x_new_reward, x_new_gt, x_new_pred = \
                    #     self.generate_new_feature_vector(model, user.flatten(), self.users_gt[index_in_users], self.users_preds[index_in_users], 
                    #         role_model, role_model_gt, role_model_pred, 
                    #         1, optimizer, 'effort', 
                    #         'reward', delta)
                    # sens_efforts.append(x_new_effort)
                    
                    dir_up_cols_generator = cf.get_up_cols(exp.dataset_info[self.dataset]['variable_constraints'], self.feature_info)
                    for dir_up_cols in dir_up_cols_generator:
                        assert np.all(role_model[dir_up_cols] >= user.flatten()[dir_up_cols]) # sanity check

                for i, user in enumerate(filtered_users_nosens):
                    print("Computing for user", np.where(sub_filter_nosens)[0][i])
                    user = np.array([user])
                    index_in_users = np.where(sub_filter_nosens)[0][i]

                    # optimizer = ge.SamplingMethod(np.array([0]), self.feature_info, self.cost_funcs, self.cost_funcs_rev, 
                    #     exp.dataset_info[self.dataset]['variable_constraints'], model, self.dataset)
                    # role_model, role_model_gt, role_model_pred = optimizer.sampling_based_explanations(
                    #     user, 
                    #     self.role_model_users, 
                    #     self.role_model_users_gt_labels,
                    #     self.role_model_users_pred,
                    #     user_gt_labels_nosens[i],
                    #     user_predicted_labels_nosens[i],
                    #     user_sens_group=0,
                    #     return_only_user=True)

                    role_model, role_model_effort, role_model_reward, role_model_utility = \
                        self.sampling_based_explanations(
                            user, 
                            self.role_model_users, 
                            self.role_model_users_gt_labels,
                            self.role_model_users_pred,
                            user_gt_labels_nosens[i],
                            user_predicted_labels_nosens[i],
                            user_sens_group=0,
                            cost_sens_group=np.array([0]),
                            variable_to_optimize='effort',
                            variable_to_threshold='reward',
                            threshold_value=delta
                        )
                    nosens_efforts.append(role_model_effort)

                    # x_new, x_new_utility, x_new_effort, x_new_reward, x_new_gt, x_new_pred = \
                    #     self.generate_new_feature_vector(model, user.flatten(), self.users_gt[index_in_users], self.users_preds[index_in_users], 
                    #         role_model, role_model_gt, role_model_pred, 
                    #         0, optimizer, 'effort', 
                    #         'reward', delta)
                    # nosens_efforts.append(x_new_effort)

                    dir_up_cols_generator = cf.get_up_cols(exp.dataset_info[self.dataset]['variable_constraints'], self.feature_info)
                    for dir_up_cols in dir_up_cols_generator:
                        assert np.all(role_model[dir_up_cols] >= user.flatten()[dir_up_cols]) # sanity check
                sens_effort_with_reward.append(np.nanmean(sens_efforts))
                nosens_effort_with_reward.append(np.nanmean(nosens_efforts))

            model_to_effort_sens[model] = [self.reward_deltas, sens_effort_with_reward]
            model_to_effort_nosens[model] = [self.reward_deltas, nosens_effort_with_reward]

            with open(self.res_file_path, 'a') as res_file:
                res_file.write("== {} ==\n\n".format(str(model)))
                res_file.write("{}\n\n{}\n\n{}\n\n".format(
                    aeio.plot_one_var_vs_other(self.res_dir, model, self.effort_deltas, sens_utils_with_effort, nosens_utils_with_effort, 'Effort', 'Average Utility'),
                    aeio.plot_one_var_vs_other(self.res_dir, model, self.effort_deltas, sens_reward_with_effort, nosens_reward_with_effort, 'Effort', 'Average Reward'),
                    aeio.plot_one_var_vs_other(self.res_dir, model, self.reward_deltas, sens_effort_with_reward, nosens_effort_with_reward, 'Reward', 'Average Effort')
                    ))
        with open(self.res_file_path, 'a') as res_file:
            res_file.write("== All Models in One ==\n\n".format(str(model)))
            res_file.write("{}\n\n{}\n\n{}\n\n".format(
                aeio.plot_one_var_vs_other_together(self.res_dir, model_to_utility_sens, model_to_utility_nosens, 'Effort', 'Average Utility'),
                aeio.plot_one_var_vs_other_together(self.res_dir, model_to_reward_sens, model_to_reward_nosens, 'Effort', 'Average Reward'),
                aeio.plot_one_var_vs_other_together(self.res_dir, model_to_effort_sens, model_to_effort_nosens, 'Reward', 'Average Effort')
            ))
        out.upload_results([self.res_dir + '/disparity_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.png')
        out.upload_results([self.res_dir + '/disparity_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.pdf')
        out.create_dir(self.res_dir + '/plots_pickled_data')
        joblib.dump(model_to_utility_sens, self.res_dir + '/plots_pickled_data/model_to_utility_sens.pkl' if not exp.FAIRNESS_CONSTRAINTS else self.res_dir + '/plots_pickled_data/model_to_utility_sens_fc.pkl')
        joblib.dump(model_to_utility_nosens, self.res_dir + '/plots_pickled_data/model_to_utility_nosens.pkl' if not exp.FAIRNESS_CONSTRAINTS else self.res_dir + '/plots_pickled_data/model_to_utility_nosens_fc.pkl')
        joblib.dump(model_to_reward_sens, self.res_dir + '/plots_pickled_data/model_to_reward_sens.pkl' if not exp.FAIRNESS_CONSTRAINTS else self.res_dir + '/plots_pickled_data/model_to_reward_sens_fc.pkl')
        joblib.dump(model_to_reward_nosens, self.res_dir + '/plots_pickled_data/model_to_reward_nosens.pkl' if not exp.FAIRNESS_CONSTRAINTS else self.res_dir + '/plots_pickled_data/model_to_reward_nosens_fc.pkl')
        joblib.dump(model_to_effort_nosens, self.res_dir + '/plots_pickled_data/model_to_effort_nosens.pkl' if not exp.FAIRNESS_CONSTRAINTS else self.res_dir + '/plots_pickled_data/model_to_effort_nosens_fc.pkl')
        joblib.dump(model_to_effort_sens, self.res_dir + '/plots_pickled_data/model_to_effort_sens.pkl' if not exp.FAIRNESS_CONSTRAINTS else self.res_dir + '/plots_pickled_data/model_to_effort_sens_fc.pkl')

if __name__=='__main__':
    try:
        test_or_train = str(sys.argv[1].lower().strip())
    except:
        raise ValueError("Usage: python effort_reward_function_plots.py train or python experiment.py train")
    start = time.time()
    erp = EffortRewardPlots()
    erp.run(test_or_train)
    end = time.time()
    print ("Time taken: {:.3f}".format(end - start))