### Use this to generate role model based explanations for the first hop.
import numpy as np
import pandas as pd
import time
import sys, os
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
# from datasets import data_util as du

CLASSIFICATION = 'classification'
REGRESSION = 'regression'
# Only set this to True if you have trained a fairness constraints model for the particular dataset you are evaluating and have put the weights file (.mat) in the root folder
FAIRNESS_CONSTRAINTS = False

dataset_info = {
    'GermanCredit': {'cost_funcs': cf.get_german_credit_cost_funcs, 'sens_f': ('men', 'women', 'sex_male')},
    'Adult': {'cost_funcs': cf.get_generic_cost_funcs, 'sens_f': ('men', 'women', 'sex_Male'), 'variable_constraints': cf.GERMAN_CREDIT_DIRS},
    'CreditDefault': {'cost_funcs': cf.get_credit_default_cost_funcs, 'sens_f': ('men', 'women', 'sex_male'), 'variable_constraints': cf.CREDIT_DEFAULT_DIRS},
    'PreprocCreditCardDefault': {'cost_funcs': cf.get_pre_proc_credit_default_cost_funcs, 'sens_f': ('men', 'women', 'Male_0'), 
                                'variable_constraints': cf.PRE_PROC_CREDIT_DEFAULT_DIRS, 'prediction_task': CLASSIFICATION},
    'StudentPerf': {'cost_funcs': cf.get_student_perf_cost_funcs, 'sens_f': ('0', '1', 'sex_Male'), 
                    'variable_constraints': cf.STUDENT_PERF_DIRS, 'variable_constraints_rev': cf.STUDENT_PERF_DIRS_REV, 'prediction_task': REGRESSION},
    'StudentPerfMut': {'cost_funcs': cf.get_student_perf_cost_funcs, 'sens_f': ('0', '1', 'sex_Male'), 
                    'variable_constraints': cf.STUDENT_PERF_DIRS, 'variable_constraints_rev': cf.STUDENT_PERF_DIRS_REV, 'prediction_task': REGRESSION},
    'StudentPerfMutPlusImmut': {'cost_funcs': cf.get_student_perf_cost_funcs, 'sens_f': ('0', '1', 'sex_Male'), 
                    'variable_constraints': cf.STUDENT_PERF_DIRS, 'variable_constraints_rev': cf.STUDENT_PERF_DIRS_REV, 'prediction_task': REGRESSION},
    'CrimesCommunities': {'cost_funcs': cf.get_student_perf_cost_funcs, 'sens_f': ('0', '1', 'MajorityRaceWhite'), 
                    'variable_constraints': cf.CRIMES_DIRS, 'variable_constraints_rev': cf.CRIMES_DIRS_REV, 'prediction_task': REGRESSION}
}
seed = 12345

def base_exp(return_vars=False, test_or_train=None):
    #dataset = "GermanCredit"
    #dataset = "Adult"
    #dataset = "CreditDefault"
    #dataset = "PreprocCreditCardDefault"
    dataset = "StudentPerf"
    # dataset = "StudentPerfMut"
    # dataset = "StudentPerfMutPlusImmut"
    # dataset = "CrimesCommunities"
    if FAIRNESS_CONSTRAINTS:
        models = [lm.LinRegFC(11.9, dataset), lm.LinRegFC(12.9, dataset), lm.LinRegFC(13.9, dataset), lm.LinRegFC(14.9, dataset)]
    else:
        if dataset_info[dataset]['prediction_task'] == CLASSIFICATION:
            models = [lm.LogReg(), lm.DT(), lm.SVM(), lm.NN()]
        elif dataset_info[dataset]['prediction_task'] == REGRESSION:
            models = [lm.LinReg(), lm.NNReg(), lm.DTReg()]
            if dataset == "StudentPerfMut":
                models = [lm.RidgeReg(0.1)]
            elif dataset == "StudentPerfMutPlusImmut":
                models = [lm.RidgeReg(200)]
    if return_vars:
        return dataset, models
    evaluate_models(dataset, models, test_or_train)

def evaluate_models(dataset, models, test_or_train, subsample_size=None, num_investigation_users = None):
    """
    Main analysis function.
    Loads the selected dataset
    generates explanations for individuals and analyzes efforts.
    """

    models_to_individual_explanation_strings_sens, models_to_individual_explanation_strings_nosens = {}, {} # global mapping

    res_dir = 'results/{}'.format(dataset)
    out.create_dir(res_dir)
    res_dir = res_dir if not FAIRNESS_CONSTRAINTS else '{}/FC'.format(res_dir)
    out.create_dir(res_dir)
    res_file_path = res_dir + '/res.txt'
    wiki_parent_path = "Actionable-Explanations/Simple-Explanations-{}".format(dataset)

    sens_group_desc = dataset_info[dataset]['sens_f']
    learning_env = dec_rule_env.DecRuleEnv(dataset, sens_group_desc)
    learning_env.load_data()

    feature_info = learning_env.feature_info
    print ("\n\nfeature_info original:{}\n\n".format(learning_env.feature_info))
    x_test_original = learning_env.x_test
    y_test = (learning_env.y_test).astype(bool if dataset_info[dataset]['prediction_task'] == CLASSIFICATION else float)
    x_train_original = learning_env.x_train
    y_train = (learning_env.y_train).astype(bool if dataset_info[dataset]['prediction_task'] == CLASSIFICATION else float)
    scaler = MinMaxScaler()
    scaler.fit(x_train_original)
    x_train = scaler.transform(x_train_original)

    with open('processed_student_data.csv','w') as fp:
        pd.DataFrame(data=np.append(x_train, y_train.reshape(x_train.shape[0],1), axis=1), columns=aeio.get_feature_names(feature_info) + ['G3']).to_csv(fp, index=False)

    x_test = scaler.transform(x_test_original)
    
    sens_group = ~learning_env.x_control[sens_group_desc[-1]]
    sens_group_train = ~learning_env.x_control_train[sens_group_desc[-1]]
    sens_group_test = ~learning_env.x_control_test[sens_group_desc[-1]]
    ds_statistics = get_dataset_statistics_temp(learning_env.y, sens_group, dataset_info[dataset]['prediction_task'])
    users = np.append(x_train, x_test, axis=0)
    user_gt_labels = np.append(y_train, y_test, axis=0).astype(bool if dataset_info[dataset]['prediction_task'] == CLASSIFICATION else float)

    ##Use these for the remaining analysis; in case you want to change analysis from test to train or vice versa this is the place to change the var assignments; 
    ##also change vars `role_model_users_pred` and `users_preds`
    role_model_users = x_train # This should not change
    role_model_users_gt_labels = y_train # This should not change
    role_model_users_sens_group = sens_group_train # This should not change
    users = x_test if test_or_train == 'test' else x_train # change this based on which group's explanations are needed (test or train)
    users_sens_group = sens_group_test if test_or_train == 'test' else sens_group_train # change this based on which group's explanations are needed (test or train)
    users_gt = y_test if test_or_train == 'test' else y_train # change this based on which group's explanations are needed (test or train)

    # If not already found, search for common negative users for in-depth analysis
    common_negative_users_sens_filename = "./common_negative_users/{}/random_{}_users_sens.txt".format(dataset, test_or_train) if not FAIRNESS_CONSTRAINTS \
        else "./common_negative_users/{}/random_{}_users_sens_fc.txt".format(dataset, test_or_train)
    common_negative_users_nosens_filename = "./common_negative_users/{}/random_{}_users_nosens.txt".format(dataset, test_or_train) if not FAIRNESS_CONSTRAINTS \
                else "./common_negative_users/{}/random_{}_users_nosens_fc.txt".format(dataset, test_or_train)
    if not os.path.exists(common_negative_users_sens_filename) or not os.path.exists(common_negative_users_nosens_filename):
        out.create_dir('./common_negative_users')
        out.create_dir('./common_negative_users/{}'.format(dataset))
        overall_negative_sens, overall_negative_nosens = None, None
        for m in models:
            clf = m
            exists, loaded_clf = aeio.load_model(clf, dataset)
            if exists:
                clf = loaded_clf
                print ("Loaded {}...".format(str(clf)))
            else:
                print ("Training {}...".format(str(clf)))
                clf.train(role_model_users, role_model_users_gt_labels)
                aeio.persist_model(clf, dataset)
            y_pred = clf.predict(users).astype(bool if dataset_info[dataset]['prediction_task'] == CLASSIFICATION else float)
            m_negatives_sens = set(np.where(np.logical_and(y_pred < users_gt, users_sens_group))[0])
            m_negatives_nosens = set(np.where(np.logical_and(y_pred < users_gt, ~users_sens_group))[0])
            if overall_negative_nosens is None:
                overall_negative_nosens, overall_negative_sens = m_negatives_nosens, m_negatives_sens
            else:
                overall_negative_sens = overall_negative_sens.intersection(m_negatives_sens)
                overall_negative_nosens = overall_negative_nosens.intersection(m_negatives_nosens)
        print ("\n{}\n".format(np.array(list(overall_negative_sens))))
        print ("\n{}\n".format(np.array(list(overall_negative_nosens))))
        np.savetxt(common_negative_users_sens_filename, np.array(list(overall_negative_sens)))
        np.savetxt(common_negative_users_nosens_filename, np.array(list(overall_negative_nosens)))

    with open(res_file_path, 'w') as res_file:
        res_file.write("Sensitive group: {}\n\n{}".format(sens_group_desc[1], ds_statistics))

        # feature_desc = du.get_feature_descriptions(feature_info)
        feature_desc = get_feature_descriptions_temp(feature_info)
        res_file.write(feature_desc)

    analysis = [("group-efforts", role_model_users_sens_group)] #[("overall-efforts", None)] # Only run for individual users for now

    # TODO: keep an eye on this mapping
    cost_groups = {cf.ONE_GROUP_IND: "all", 0: sens_group_desc[0], 1: sens_group_desc[1]} # cf.ONE_GROUP_IN = -1

    for (analysis_name, cost_sens_group) in analysis:
        group_res_file_path = "{}/{}_res_{}.txt".format(res_dir, analysis_name, test_or_train)

        cost_funcs, feature_val_costs = dataset_info[dataset]['cost_funcs'](feature_info, 
            role_model_users, cost_sens_group, dataset_info[dataset]['variable_constraints'])
        cost_funcs_rev, feature_val_costs_rev = dataset_info[dataset]['cost_funcs'](feature_info, 
            role_model_users, cost_sens_group, dataset_info[dataset]['variable_constraints_rev'])

        print ("{}\n".format(feature_val_costs))
        print ("{}\n".format(cost_funcs))

        with open(group_res_file_path, 'w') as group_res_file:
            group_res_file.write("== Cost functions ==\n\nCosts are computed as (fraction < new value) - (fraction < old value), where values are ordered in the direction of increasing effort it takes to reach them.\n\n")
            for sens_group_val, costs in feature_val_costs.items(): # Code never goes into this loop for generic cost func
                sens_group_desc = cost_groups[sens_group_val]
                group_res_file.write("=== Costs for group {}: ===\n\n{}\n".format(
                    sens_group_desc, aeio.get_cost_func_desc(costs)))
                group_res_file.write("=== Reverse costs for group {}: ===\n\n{}\n".format(
                    sens_group_desc, aeio.get_cost_func_desc(feature_val_costs_rev[sens_group_val])))

        # Randomly choose num_investigation_users to check their feature
        # values and the explanations generated for them
        np.random.seed(seed)
        investigation_users_sens = np.loadtxt(common_negative_users_sens_filename).astype(int)[:num_investigation_users] \
            if num_investigation_users is not None else np.loadtxt(common_negative_users_sens_filename).astype(int)  # np.random.randint(len(filtered_users_sens), size=num_investigation_users) # Hardcoded so as to have same accross all models
        investigation_users_nosens = np.loadtxt(common_negative_users_nosens_filename).astype(int)[:num_investigation_users] \
            if num_investigation_users is not None else np.loadtxt(common_negative_users_nosens_filename).astype(int) # np.random.randint(len(filtered_users_nosens), size=num_investigation_users) # Hardcoded so as to have same accross all models
        # Values for individual users

        #print("Users:\n", group_desc)
        disparity_table_heading = ["model"]
        disparity_table_formats = [None]
        disparity_table_values = []

        with open(group_res_file_path, 'a') as group_res_file:
            group_res_file.write("=== Individual explanations: ===\n\n")
            group_res_file.write("All disparities are calculated as abs(sens_val - nosens_val)\n\n")

        for model in models:
            model_start = time.time()
            clf = model
            exists, loaded_clf = aeio.load_model(clf, dataset)
            if exists:
                clf = loaded_clf
                print ("Loaded {}...".format(str(clf)))
            else:
                print ("Training {}...".format(str(clf)))

                # TODO: ugly
                if isinstance(model, lm.FCLogReg):
                    clf.train(role_model_users, role_model_users_gt_labels, learning_env.x_control_train)
                else:
                    clf.train(role_model_users, role_model_users_gt_labels)
                aeio.persist_model(clf, dataset)

            model_end = time.time()
            performance_stats = eval_formula.eval_model(clf, users, users_gt, dataset_info[dataset]['prediction_task'])
            y_test_pred = clf.predict(users).astype(bool if dataset_info[dataset]['prediction_task'] == CLASSIFICATION else float)
            y_train_pred = clf.predict(role_model_users).astype(bool if dataset_info[dataset]['prediction_task'] == CLASSIFICATION else float)

            ####Common var names; change these if you want to change analysis from train to test or vice versa
            role_model_users_pred = y_train_pred # This should not change
            users_preds = y_test_pred if test_or_train == 'test' else y_train_pred # change this based on which group's explanations are needed (test or train)

            with open(res_file_path, 'a') as res_file:
                res_file.write("Performance of {}\n\n{}\n\n".format(str(clf), aeio.get_dict_listing(performance_stats)))
                res_file.write("Training {} took {:.2f} secs".format(str(clf), model_end - model_start))

            investigation_explanations_sens, investigation_explanations_nosens = [], []

            sub_filter_sens = np.zeros(len(users), dtype=bool)
            sub_filter_nosens = np.zeros(len(users), dtype=bool)
            sub_filter_sens[np.where(np.logical_and(users_sens_group, users_preds < users_gt))[0][:subsample_size] if \
                subsample_size is not None else np.where(np.logical_and(users_sens_group, users_preds < users_gt))[0]] = 1
            sub_filter_nosens[np.where(np.logical_and(~users_sens_group, users_preds < users_gt))[0][:subsample_size] if \
                subsample_size is not None else np.where(np.logical_and(~users_sens_group, users_preds < users_gt))[0]] = 1
            sub_filter_sens[investigation_users_sens] = 1
            sub_filter_nosens[investigation_users_nosens] = 1
            filtered_users_sens = users[sub_filter_sens]
            user_gt_labels_sens = users_gt[sub_filter_sens]
            user_predicted_labels_sens = users_preds[sub_filter_sens]
            filtered_users_nosens = users[sub_filter_nosens]
            user_gt_labels_nosens = users_gt[sub_filter_nosens]
            user_predicted_labels_nosens = users_preds[sub_filter_nosens]

            user_utility_sens, user_utility_nosens, anchor_indices_sens, anchor_indices_nosens = [], [], [], []
            fp_count_sens, fp_count_nosens = 0, 0

            all_users_flipped, all_users_flipped_labels = users.copy(), users_gt.copy()

            ind_start = time.time()

            for i, user in enumerate(filtered_users_sens):
                # if i % 100 == 0:
                print("Computing for user", np.where(sub_filter_sens)[0][i])
                user = np.array([user])
                index_in_users = np.where(sub_filter_sens)[0][i]

                optimizer = ge.SamplingMethod(np.array([1]), feature_info, cost_funcs, cost_funcs_rev, 
                    dataset_info[dataset]['variable_constraints'], clf, dataset)
                new_feature_vector, utility, effort, false_positive, role_model_gt, anchor_index = optimizer.sampling_based_explanations(
                    user, 
                    role_model_users, 
                    role_model_users_gt_labels,
                    role_model_users_pred,
                    user_gt_labels_sens[i],
                    user_predicted_labels_sens[i],
                    user_sens_group=1)
                dir_up_cols_generator = cf.get_up_cols(dataset_info[dataset]['variable_constraints'], feature_info)
                for dir_up_cols in dir_up_cols_generator:
                    assert np.all(new_feature_vector[dir_up_cols] >= user.flatten()[dir_up_cols]) # sanity check
                new_predicted_label = clf.predict([new_feature_vector])[0] if dataset_info[dataset]['prediction_task'] == REGRESSION else bool(clf.predict([new_feature_vector])[0])
                all_users_flipped[index_in_users] = new_feature_vector
                all_users_flipped_labels[index_in_users] = role_model_gt
                if false_positive:
                    fp_count_sens += 1
                tar_nec_vars = np.where(new_feature_vector != user[0])[0]
                tar_vals = new_feature_vector[tar_nec_vars]
                old_vals = user[0][tar_nec_vars]
                user_utility_sens.append(utility)
                anchor_indices_sens.append(anchor_index)
                if np.where(sub_filter_sens)[0][i] in investigation_users_sens:
                    feature_wise_effort, explanation = aeio.get_feature_wise_effort(feature_info, user[0], tar_nec_vars, tar_vals, cost_funcs, cost_funcs_rev, 
                        True, role_model_gt, new_predicted_label, user_gt_labels_sens[i], user_predicted_labels_sens[i])    
                    individual_feature_costs_str = aeio.get_feature_wise_str(feature_wise_effort, explanation)
                    user_explanation = \
                        " * User {} explanation, utility: {:.3f}, effort: {:.3f}\n{}\n  * User gt label: {}, user_predicted_label:{}, role model gt label: {}, role model predicted label: {}; explanation:\n{}\n  * Old feature vals for user {}:\n{}\n".format(
                            np.where(np.where(sub_filter_sens)[0][i] == investigation_users_sens)[0][0], utility, effort, individual_feature_costs_str, 
                            user_gt_labels_sens[i], user_predicted_labels_sens[i], role_model_gt, new_predicted_label,
                            aeio.get_conditions_str(feature_info, tar_nec_vars, tar_vals, scaler=scaler, level=3), np.where(np.where(sub_filter_sens)[0][i] == investigation_users_sens)[0][0], 
                            aeio.get_conditions_str(feature_info, tar_nec_vars, old_vals, scaler=scaler, level=3))
                    investigation_explanations_sens.append(user_explanation)

            for i, user in enumerate(filtered_users_nosens):
                # if i % 100 == 0:
                print("Computing for user", np.where(sub_filter_nosens)[0][i])
                user = np.array([user])
                index_in_users = np.where(sub_filter_nosens)[0][i]

                optimizer = ge.SamplingMethod(np.array([0]), feature_info, cost_funcs, cost_funcs_rev, 
                    dataset_info[dataset]['variable_constraints'], clf, dataset)
                new_feature_vector, utility, effort, false_positive, role_model_gt, anchor_index = optimizer.sampling_based_explanations(
                    user, 
                    role_model_users,
                    role_model_users_gt_labels,
                    role_model_users_pred,
                    user_gt_labels_nosens[i],
                    user_predicted_labels_nosens[i],
                    user_sens_group=0)
                dir_up_cols_generator = cf.get_up_cols(dataset_info[dataset]['variable_constraints'], feature_info)
                for dir_up_cols in dir_up_cols_generator:
                    assert np.all(new_feature_vector[dir_up_cols] >= user.flatten()[dir_up_cols]) # education
                new_predicted_label = clf.predict([new_feature_vector])[0] if dataset_info[dataset]['prediction_task'] == REGRESSION else bool(clf.predict([new_feature_vector])[0])
                all_users_flipped[index_in_users] = new_feature_vector
                all_users_flipped_labels[index_in_users] = role_model_gt
                if false_positive:
                    fp_count_nosens += 1
                tar_nec_vars = np.where(new_feature_vector != user[0])[0]
                tar_vals = new_feature_vector[tar_nec_vars]
                old_vals = user[0][tar_nec_vars]
                user_utility_nosens.append(utility)
                anchor_indices_nosens.append(anchor_index)
                if np.where(sub_filter_nosens)[0][i] in investigation_users_nosens:
                    feature_wise_effort, explanation = aeio.get_feature_wise_effort(feature_info, user[0], tar_nec_vars, tar_vals, cost_funcs, cost_funcs_rev, 
                        False, role_model_gt, new_predicted_label, user_gt_labels_nosens[i], user_predicted_labels_nosens[i])    
                    individual_feature_costs_str = aeio.get_feature_wise_str(feature_wise_effort, explanation)
                    user_explanation = \
                        " * User {} explanation, utility: {:.3f}, effort: {:.3f}\n{}\n  * User gt label: {}, user_predicted_label:{}, role model gt label: {}, role model predicted label: {}; explanation:\n{}\n  * Old feature vals for user {}:\n{}\n".format(
                            np.where(np.where(sub_filter_nosens)[0][i] == investigation_users_nosens)[0][0], utility, effort, individual_feature_costs_str, 
                            user_gt_labels_nosens[i], user_predicted_labels_nosens[i], role_model_gt, new_predicted_label,
                            aeio.get_conditions_str(feature_info, tar_nec_vars, tar_vals, scaler=scaler, level=3), np.where(np.where(sub_filter_nosens)[0][i] == investigation_users_nosens)[0][0], 
                            aeio.get_conditions_str(feature_info, tar_nec_vars, old_vals, scaler=scaler, level=3))
                    investigation_explanations_nosens.append(user_explanation)

            user_utility_sens, user_utility_nosens, anchor_indices_sens, anchor_indices_nosens = \
                np.array(user_utility_sens), np.array(user_utility_nosens), np.array(anchor_indices_sens), np.array(anchor_indices_nosens)

            summary_of_useful_explanations = "Total explanations given: {} ({} sens group, {} non-sens group); Useful explanations (utility > 0): {} ({} sens group, {} non-sens group".format(
                len(filtered_users_sens) + len(filtered_users_nosens), len(filtered_users_sens), len(filtered_users_nosens),
                np.count_nonzero(user_utility_sens > 0) + np.count_nonzero(user_utility_nosens > 0), np.count_nonzero(user_utility_sens > 0), np.count_nonzero(user_utility_nosens > 0))

            all_users_utilities = np.zeros(len(users))
            all_users_utilities[sub_filter_sens] = user_utility_sens
            all_users_utilities[sub_filter_nosens] = user_utility_nosens
            all_user_anchors = np.full(len(users), fill_value=np.nan)
            all_user_anchors[sub_filter_sens] = anchor_indices_sens
            all_user_anchors[sub_filter_nosens] = anchor_indices_nosens
            out.create_dir('./flipped_datasets')
            out.create_dir('./flipped_datasets/{}'.format(dataset))
            np.savetxt('./flipped_datasets/{}/{}_all_x_{}.txt'.format(dataset, clf.filename(), test_or_train), all_users_flipped)
            np.savetxt('./flipped_datasets/{}/{}_all_y_{}.txt'.format(dataset, clf.filename(), test_or_train), all_users_flipped_labels)
            np.savetxt('./flipped_datasets/{}/{}_utilities_{}.txt'.format(dataset, clf.filename(), test_or_train), all_users_utilities)
            np.savetxt('./flipped_datasets/{}/{}_anchors_{}.txt'.format(dataset, clf.filename(), test_or_train), all_user_anchors)
            
            # group_sens = users[investigation_users_sens]
            # group_efforts_sens = user_utility_sens[investigation_users_sens]
            user_explanations_sens = "".join(string for string in investigation_explanations_sens)
            models_to_individual_explanation_strings_sens[str(clf)] = user_explanations_sens

            # group_nosens = users[investigation_users_nosens]
            # group_efforts_nosens = user_utility_nosens[investigation_users_nosens]
            user_explanations_nosens = "".join(string for string in investigation_explanations_nosens)
            models_to_individual_explanation_strings_nosens[str(clf)] = user_explanations_nosens

            ind_end = time.time()

            if len(disparity_table_heading) <= 1:
                heading, formats, values = eval_formula.get_disparity_measures(users_gt, users_preds, users_sens_group, 
                                np.nanmean(user_utility_sens), np.nanmean(user_utility_nosens), dataset_info[dataset]['prediction_task'], return_heading_and_formats=True)
                disparity_table_heading += heading
                disparity_table_formats += formats
            else:
                values = eval_formula.get_disparity_measures(users_gt, users_preds, users_sens_group, 
                                np.nanmean(user_utility_sens), np.nanmean(user_utility_nosens), dataset_info[dataset]['prediction_task'], return_heading_and_formats=False)
            disparity_table_values.append([str(clf)] + values)

            with open(group_res_file_path, 'a') as group_res_file:
                group_res_file.write("  * For {}:\n\n".format(str(model)))
                group_res_file.write("   * Computation of explanations, for {} ({} sens, {} nonsens) \
                    individual users took {:.2f} seconds.\n\n   * # False positive role models = {} sens, {} \
                    nonsens.\n\n".format(len(filtered_users_sens) + len(filtered_users_nosens), 
                        len(filtered_users_sens), len(filtered_users_nosens), ind_end - ind_start, 
                        fp_count_sens, fp_count_nosens))
                group_res_file.write("   * {}\n\n".format(summary_of_useful_explanations))

        with open(group_res_file_path, 'a') as group_res_file:
            group_res_file.write("Disparity measures of different models:\n\n{}\n".format(out.get_table(disparity_table_heading, 
                disparity_table_values, val_format=disparity_table_formats)))
            group_res_file.write(aeio.get_disparity_plots(res_dir, disparity_table_heading, disparity_table_values, 
                filename='all_disp_in_one_{}'.format(test_or_train)))
            # group_res_file.write("Effort statistics for sens group:\n\n{}\n".format(ind_effort_stats_sens))
            # group_res_file.write("Effort statistics for non-sens group:\n\n{}\n".format(ind_effort_stats_nosens))
            # group_res_file.write("Top explanations for sens group:\n\n{}\n".format(top_explanations_sens))
            # group_res_file.write("Top explanations for non-sens group:\n\n{}\n".format(top_explanations_nosens))
            if num_investigation_users is None or num_investigation_users > 0:
            #     group_res_file.write("User explanation examples for randomly selected sensitive users:\n\n{}\nExplanations for sensitive users in group:\n\n{}\n".format(group_desc_sens, user_explanations_sens))
                for k,v in models_to_individual_explanation_strings_sens.items():
                    group_res_file.write("Randomly chosen sens users' explanations for {}:\n\n{}\n\n".format(k,v))
                for k,v in models_to_individual_explanation_strings_nosens.items():
                    group_res_file.write("Randomly chosen nonsens users' explanations for {}:\n\n{}\n\n".format(k,v))
            # group_res_file.write("== IGNORE STUFF BELOW THIS FOR NOW ==\n\n")
            # group_res_file.write("=== Group Explanations ===\n\n")

    # out.upload_results([res_dir], 'results', aeio.SERVER_PROJECT_PATH, '.png')
    # out.upload_results([res_dir + '/disparity_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.png')
    # out.upload_results([res_dir], 'results', aeio.SERVER_PROJECT_PATH, '.pdf')
    # out.upload_results([res_dir + '/disparity_plots'], 'results', aeio.SERVER_PROJECT_PATH, '.pdf')

def get_dataset_statistics_temp(y, sens_group, prediction_task):
    if prediction_task == REGRESSION:
        ds_statistics = '''Dataset Statistics:\n\n * number of instances: %d\n * with sensitive feature: %.3f \n'''
        number_instances = len(y)
        sensitive_instances = (np.count_nonzero(sens_group)/len(sens_group))
        return (ds_statistics % (number_instances, sensitive_instances))
    elif prediction_task == CLASSIFICATION:
        ds_statistics = '''Dataset Statistics:\n\n * number of instances: %d\n
 * labeled positive: %.3f \n
 * with sensitive feature: %.3f \n
 * percentage of positively labeled people with sensitive feature: %.3f \n
 * percentage of negatively labeled people with sensitive feature: %.3f \n'''
        number_instances = len(y)
        positive_instances = (np.count_nonzero(y)/len(y))
        sensitive_instances = (np.count_nonzero(sens_group)/len(sens_group))
        percentage_pos_with_sens = (np.count_nonzero(np.logical_and(y, sens_group))/np.count_nonzero(y))
        percentage_neg_with_sens = (np.count_nonzero(np.logical_and(~(y.astype('bool')), sens_group))/np.count_nonzero(~(y.astype('bool'))))
        return (ds_statistics % (number_instances, positive_instances, sensitive_instances, percentage_pos_with_sens, percentage_neg_with_sens))

def get_feature_descriptions_temp(feature_info):
    feature_desc = "\nFeatures (only used ones):\n\n"
    for ft_info in feature_info:
        feature_desc += " * {}: {}, values: \n".format(ft_info[0], 'Binary' if ft_info[1] == 'cat_bin' else "Continuous")
        # for i in range(len(ft_info[2])):
        #     feature_desc += ("{}\n" if i == len(ft_info[2]) - 1 else "{}, ").format(ft_info[2][i][1])
    feature_desc += "\n"
    return feature_desc


def main(test_or_train):
    start = time.time()

    base_exp(test_or_train=test_or_train)
    #fairness_constraint_exp()

    #synth_exp.synth_2v_exp()
    #synth_exp.synth_3v_exp()

    #credit_default_check()

    end = time.time()
    print("executed for {:.2f} seconds".format(end - start))

if __name__ == "__main__":
    try:
        test_or_train = str(sys.argv[1].lower().strip())
        main(test_or_train)
    except Exception as e:
        raise ValueError("Usage: python experiment.py train or python experiment.py train")
