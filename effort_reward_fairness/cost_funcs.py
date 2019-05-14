import numpy as np
import sys
from collections import defaultdict
from itertools import repeat

sys.path.append('../util/')
from datasets import file_util as fu

ONE_GROUP_IND = -1

class DistCost(object):
    """Cost function that assigns cost based on by how many steps
    a value changed
    """
    def __init__(self, weight=1):
        self.weight = weight

    def __call__(self, old_index, old_val, new_index, new_val, sens_group=None):
        return self.weight * abs(old_index - new_index)

class UnidirCost(object):
    """Unidirectional cost function wrapper that takes a cost
    function and only allows changes in one direction.
    Changes in the other direction have a -1 cost which means that
    the corresponding clause will be hard.
    """
    def __init__(self, cost_func, increasing=True):
        self.increasing = increasing
        self.cost_func = cost_func

    def __call__(self, old_index, old_val, new_index, new_val):
        if self.increasing and (new_index < old_index):
            return -1
        elif not self.increasing and (new_index > old_index):
            return -1
        else:
            return self.cost_func(old_index, old_val, new_index, new_val)

class ImmutableCost(object):
    """Assignes -1/infinite to a feature, making it immutable"""
    def __call__(self, old_val, new_val, sens_group=None):
        return -1

class FracCost(object):
    """Fractional cost function that takes into account how many
    users have certain feature values and yields increasingly high
    costs for moving to higher-effort feature values possessed by
    fewer users.
    Associates zero cost with moving to a lower-effort feature value
    """
    def __init__(self, costs):
        self.costs = costs

    def __str__(self):
        return "{}".format(self.costs)

    def __call__(self, old_index, old_val, new_index, new_val,
            sens_group=ONE_GROUP_IND):
        old_cost = self.costs[sens_group][old_val]
        new_cost = self.costs[sens_group][new_val]
        cost = (new_cost - old_cost)
        return cost

class ContCost(object):
    '''FracCost but for cont features'''
    def __init__(self, sens_to_data_vector, direction):
        self.sens_to_data_vector = sens_to_data_vector
        # print ("\n{}\n".format(sens_to_data_vector))
        self.direction = direction

    def __call__(self, old_val, new_val, sens_group=ONE_GROUP_IND):
        if self.direction == EXP_HIGHER:
            old_cost = np.count_nonzero(self.sens_to_data_vector[sens_group] < float(old_val))/len(self.sens_to_data_vector[sens_group])
            new_cost = np.count_nonzero(self.sens_to_data_vector[sens_group] < float(new_val))/len(self.sens_to_data_vector[sens_group])
            return (new_cost - old_cost)
        else:
            old_cost = np.count_nonzero(self.sens_to_data_vector[sens_group] > float(old_val))/len(self.sens_to_data_vector[sens_group])
            new_cost = np.count_nonzero(self.sens_to_data_vector[sens_group] > float(new_val))/len(self.sens_to_data_vector[sens_group])
            return (new_cost - old_cost)


def compute_efforts(feature_infos, cost_funcs, cost_funcs_rev, group, explanation, sens_group=None):
    """Computes the efforts for each user in a group that they need
    to invest if they change their features according to a given
    explanation.

    feature_info -- a list with information about names, types and values
        of all the features in the dataset.
    cost_funcs -- a dictionary mapping feature names to cost functions
    group -- the group of users to compute efforts for
    explanation -- a dictionary from feature names to target indices and values that describes for the features that need to be changed which index of the binarized version they need to be changed to as well as value corresponding to that index.
    sens_group -- a vector incoding membership in the sensitive group,
        used for sensitive group specific cost functions (default None)

    Returns a list of efforts, one for each user in the group
    """
    if sens_group is None:
        sens_group = repeat(ONE_GROUP_IND)
    efforts = []
    for user, user_sens_group in zip(group, sens_group):
        cur_user_inds, cur_user_vals = bin_to_index_vals(feature_infos, user)
        # print ("\n{}, {}\n".format(cur_user_inds, cur_user_vals))
        user_cost = 0
        for (fname, ftype, _), cur_ind, cur_val in zip(feature_infos, cur_user_inds, cur_user_vals):
            (new_ind, new_val) = explanation[fname] if fname in explanation else (cur_ind, cur_val)
            # print ("\n\nsens_group: {}\n\n".format(user_sens_group))
            # print ("feature_name: {}".format(fname))
            # print ("feature_infos: {}".format(len(feature_infos)))
            # print ("cost_func: {}".format(cost_funcs[fname]))
            # print ("cost_func_rev: {}".format(cost_funcs_rev[fname]))
            # print ("\n\ncur_ind: {}, cur_val: {}, new_ind: {}, new_val: {}\n\n".format(cur_ind, cur_val, new_ind, new_val))
            feature_cost_forward = cost_funcs[fname](cur_val, new_val, user_sens_group)
            feature_cost_rev = cost_funcs_rev[fname](cur_val, new_val, user_sens_group)
            feature_cost = np.max((0, max(feature_cost_forward, feature_cost_rev))) # might be -1 because of immutable features
            # if ftype == fu.ATTR_CONT:
            #     print (fname, cur_val, new_val, feature_cost, feature_cost_forward, feature_cost_rev)
            user_cost += feature_cost/float(len(feature_infos))
        efforts.append(user_cost)
    return efforts


def bin_to_index_vals(feature_info, user, cont_permitted=True):
    """Converts a binary user feature vector to an array containing
    the source feature index as well as the source feature value
    for each source feature, for which there might exist multiple
    binary feature.
    Effectively reverses one-hot encoding.
    """

    user_indices = []
    user_vals = []

    ind = 0
    for fname, ftype, vlabels in feature_info:
        if ftype == fu.ATTR_CONT or ftype == fu.ATTR_CAT_BIN: # here binary is being treated just like cont
            assert cont_permitted
            assert len(vlabels) == 1
            user_indices.append(ind)
            user_vals.append(str(user[ind]))
        elif ftype == fu.ATTR_CAT_BIN:
            uval = user[ind]
            findex, fval = get_bin_index_val(vlabels, uval)
            user_indices.append(findex)
            user_vals.append(fval)
        else: #MULT or CONT_QUANT
            feature_vals = user[ind:ind + len(vlabels)]
            assert np.sum(feature_vals) == 1, "not exactly one value true for feautre {}, labels: {}, values: {}".format(fname, str(vlabels), str(feature_vals))
            val_index = np.nonzero(feature_vals)[0][0]
            user_indices.append(val_index)
            vres = vlabels[val_index]
            if isinstance(vres, tuple):
                _, val_label = vres
            else:
                val_label = vres
            user_vals.append(val_label)

        ind += len(vlabels)

    return user_indices, user_vals

def compute_benefit(true_val, predicted_val, sens_group):
    """compute the benefit function as follows.
    true_val is a numpy array of ground truth
    predicted_val is a numpy array of predicted labels
    sens_group is a numpy array of sensitive group membership
    """
    if isinstance(true_val, np.ndarray) or isinstance(predicted_val, np.ndarray) or isinstance(sens_group, np.ndarray):
        true_val = true_val.astype(float)
        predicted_val = predicted_val.astype(float)
        sens_group = sens_group.astype(float)
    if isinstance(true_val, bool) or isinstance(predicted_val, bool) or isinstance(sens_group, bool) or \
            isinstance(true_val, np.bool_) or isinstance(predicted_val, np.bool_) or isinstance(sens_group, np.bool_):
        true_val, predicted_val, sens_group = int(true_val), int(predicted_val), int(sens_group)

    return predicted_val
    # return (predicted_val - true_val + 100.)

    # return (predicted_val * (true_val + 1) + (1 - true_val))/40.

    # b_11 = 0.5
    # b_10 = 0
    # b_01 = 1
    # b_00 = 0.1

    # c_1 = b_01 - b_00
    # c_2 = b_00 + b_11 - b_10 - b_01
    # c_3 = b_10 - b_00
    # c_4 = b_00

    
    # B = (c_1 * predicted_val + c_2 * true_val * predicted_val + c_3 * true_val + c_4) * (sens_group) + (predicted_val - true_val + 1.5) * (1 - sens_group)
    # return B

def compute_reward(new_true_val, new_pred_val, old_true_val, old_pred_val, sens_group):
    """
    params are all numpy arrays of same dimension
    old_true_val = stacked numpy array of user's true label
    old_pred_val = stacked numpy array of user's pred label
    new_true_val = numpy array of candidate set of role models' ground truth
    new_pred_val = numpy array of candidate set of role models' predicted val
    sens_group = stacked numpy array of user's sens group membership
    """
    return compute_benefit(new_true_val, new_pred_val, sens_group) - compute_benefit(old_true_val, old_pred_val, sens_group)

def compute_utility(new_true_val, new_pred_val, old_true_val, old_pred_val, sens_group, effort):
    """
    params are all numpy arrays of same dimension
    old_true_val = stacked numpy array of user's true label
    old_pred_val = stacked numpy array of user's pred label
    new_true_val = numpy array of candidate set of role models' ground truth
    new_pred_val = numpy array of candidate set of role models' predicted val
    sens_group = stacked numpy array of user's sens group membership
    effort = numpy array of user's effort to change to each of the candidate set of role models
    """
    reward = compute_reward(new_true_val, new_pred_val, old_true_val, old_pred_val, sens_group)
    utility = reward - effort
    return utility

def get_bin_index_val(fvals, is_pos):
    assert len(fvals) == 1
    index = 0 if is_pos else 1
    if isinstance(fvals[0], tuple) or isinstance(fvals[0], list):
        val = fvals[0][1] if is_pos else "non-" + fvals[0][1]
    else:
        val = fvals[0] if is_pos else "non-" + fvals[0]
    return index, val

def get_generic_cost_funcs(feature_info, data, sens_group=None):
    """Returns generic cost functions, that assign costs based on by how many steps a feature changed"""
    return defaultdict(DistCost), defaultdict(lambda _: defaultdict(int))

def get_immutable_cols(variable_constraints, feature_info):
    idx = 0
    desired_cols = []
    for fname, ftype, fvals in feature_info:
        if variable_constraints[fname][0] == DIR_IMMUT:
            for i in range(len(fvals)):
                desired_cols.append(idx + i)
        idx += len(fvals)
    return np.array(desired_cols)

def get_up_cols(variable_constraints, feature_info):
    idx = 0
    for fname, ftype, fvals in feature_info:
        desired_cols = []
        if variable_constraints[fname][0] == DIR_UP:
            for i in range(len(fvals)):
                desired_cols.append(idx + i)
            yield np.array(desired_cols)
        idx += len(fvals)

# Constants specifying the allowed directions of feature change
# Only changes from lower- to higher-effort feature values allowed
DIR_UP = 'up'
# Changes in both effort directions allowed
DIR_BOTH = 'both'
# Immutable feature, no changes allowed
DIR_IMMUT = 'immut'
# Not specified, i.e. feature hasn't been used in a rule so far so
# it wasn't necessary to specify costs for it
DIR_UNSPEC = 'unspec'

# Cost directions for continous features
# Higher feature values require more effort/are more expensive
EXP_HIGHER = 'higher'
# Lower feature values require more effort/are more expensive
EXP_LOWER = 'lower'

def get_dataset_cost_funcs(feature_cost_spec, feature_info, data,
        sens_group=None):
    """Compute the costs based on population fractions at, above
    or below thresholds
    
    feature_cost_spec -- dictionary that specifies for each feature
        the directionality of allowed change and the way the values
        need to be ordered to sort them in increasing order of effort
    feature_info -- a list with information about names, types and values
        of all the features in the dataset.
    data -- the dataset, i.e. the users
    sens_group -- a vector incoding membership in the sensitive group,
        used for sensitive group specific cost functions (default None)
    """

    # if no sensitive group-membership is specified then everyone is put into one default sensitive group
    if sens_group is None:
        sens_vals = [ONE_GROUP_IND]
        sens_group = np.ones(len(data)) * ONE_GROUP_IND
    else:
        sens_vals = np.unique(sens_group).astype(int)

    cost_funcs = {}
    feature_costs = {sv: {} for sv in sens_vals}

    index = 0
    for fname, ftype, fvals in feature_info:
        directionality, ordering = feature_cost_spec[fname]
        assert directionality != DIR_UNSPEC

        if directionality == DIR_IMMUT:
            for sens_val in sens_vals:
                feature_costs[sens_val][fname] = {'immutable': 0}
            cost_funcs[fname] = ImmutableCost()
            index += len(fvals)
            continue

        if ftype == fu.ATTR_CONT or ftype == fu.ATTR_CAT_BIN:
            sens_group_to_vec = {sv: [] for sv in sens_vals}
            for sens_val in sens_vals:
                if sens_val == 1:
                    feature_vector = data[:,index][sens_group]
                elif sens_val == 0:
                    feature_vector = data[:,index][~sens_group]
                else:
                    feature_vector = data[:,index]
                sens_group_to_vec[sens_val] = feature_vector
                feature_costs[sens_val][fname] = {'continuous_cost': 0} # this is just a placeholder; not the actual cost
            cost_funcs[fname] = ContCost(sens_group_to_vec, ordering)
            index += len(fvals)
            continue
            
        # Compute costs separately for each sensitive group
        # I.e. based only on the fraction of people in the sensitive
        # group relative to a threshold, not based on the fraction
        # of all users
        sens_val_costs = {}
        print (sens_vals)
        for sens_val in sens_vals:
            print (fname, ftype, fvals)
            values = data[:, index:index + len(fvals)]
            labels = [l for _, l in fvals]

            # if ftype == fu.ATTR_CONT_QUANT:
            if ordering == EXP_HIGHER:
                ordering = labels
            elif ordering == EXP_LOWER:
                ordering = labels[::-1]

            num_total = len(data[sens_group == sens_val,:])
            num_below = 0
            costs = {}

            assert set(ordering) == set(labels)

            # add costs for values in increasing order of
            # expensiveness
            for label in ordering:
                below_frac = num_below / num_total
                # above_frac = 1 - below_frac
                # if above_frac == 0:
                #     costs[label] = -1 # hard clause, it's impossible to achieve this value
                # else:
                #     costs[label] = below_frac / above_frac
                costs[label] = below_frac

                data_index = labels.index(label)
                value_count = np.count_nonzero(values[:,data_index][sens_group==sens_val])
                num_below += value_count

            assert num_below == num_total
            sens_val_costs[sens_val] = costs
            feature_costs[sens_val][fname] = costs

        cost_funcs[fname] = FracCost(sens_val_costs)
        index += len(fvals)

    return cost_funcs, feature_costs

# Cost function semantics (directionality of allowed change, direction of expensiveness) for the features in the German Credit dataset
GERMAN_CREDIT_DIRS = {
    #'checking_acc_status': (DIR_BOTH, ['<0DM', '[0,200)DM', '>=200DM', 'None']),
    'checking_acc_status': (DIR_IMMUT, ['<0DM', '[0,200)DM', '>=200DM', 'None']),
    'duration': (DIR_BOTH, EXP_LOWER),
    'credit_hist': (DIR_UNSPEC, None),
    'purpose': (DIR_IMMUT, None),
    'credit_amount': (DIR_IMMUT, EXP_LOWER),
    'savings_acc': (DIR_BOTH, ['unknown/none', '<100DM', '[100,500)DM', '[500,1000)DM', '>=1000DM']),
    'employment': (DIR_IMMUT, None),
    'installment_rate': (DIR_BOTH, EXP_HIGHER),
    'sex': (DIR_IMMUT, None),
    'debtors_guarantors': (DIR_UNSPEC, None),
    'residence_duration': (DIR_UP, EXP_HIGHER),
    'property': (DIR_IMMUT, None),
    'age': (DIR_UP, EXP_HIGHER),
    'installment_plans': (DIR_UNSPEC, None),
    'housing': (DIR_IMMUT, None),
    'num_credit_cards': (DIR_BOTH, EXP_LOWER),
    'job': (DIR_IMMUT, None),
    'num_people_liable_to': (DIR_IMMUT, None),
    'telephone': (DIR_IMMUT, None),
    'foreign_worker': (DIR_IMMUT, None)
}

def get_german_credit_cost_funcs(feature_info, data, sens_group=None):
    """Returns fractional cost functions taking into account the semantics of the German Credit dataset"""
    return get_dataset_cost_funcs(GERMAN_CREDIT_DIRS, feature_info, data, sens_group)

# Cost function semantics (directionality of allowed change, direction of expensiveness) for the features in the Credit Card Default Dataset
CREDIT_DEFAULT_DIRS = {
    'limit_bal': (DIR_BOTH, EXP_LOWER),
    'sex': (DIR_IMMUT, None),
    'education': (DIR_UP, ['unknown', 'others', 'high-school', 'university', 'graduate-school']),
    'marriage': (DIR_IMMUT, None),
    'age': (DIR_UP, EXP_HIGHER),
    'pay_1': (DIR_BOTH, EXP_LOWER),
    'pay_2': (DIR_BOTH, EXP_LOWER),
    'pay_3': (DIR_BOTH, EXP_LOWER),
    'pay_4': (DIR_BOTH, EXP_LOWER),
    'pay_5': (DIR_BOTH, EXP_LOWER),
    'pay_6': (DIR_BOTH, EXP_LOWER),
    'months_w_payment_delay': (DIR_BOTH, EXP_LOWER),
    'frac_paid_1': (DIR_BOTH, EXP_HIGHER),
    'frac_paid_2': (DIR_BOTH, EXP_HIGHER),
    'frac_paid_3': (DIR_BOTH, EXP_HIGHER),
    'frac_paid_4': (DIR_BOTH, EXP_HIGHER),
    'frac_paid_5': (DIR_BOTH, EXP_HIGHER),
    'frac_paid_6': (DIR_BOTH, EXP_HIGHER),
    #'bill_amt1': (DIR_BOTH, EXP_LOWER),
    #'bill_amt2': (DIR_BOTH, EXP_LOWER),
    #'bill_amt3': (DIR_BOTH, EXP_LOWER),
    #'bill_amt4': (DIR_BOTH, EXP_LOWER),
    #'bill_amt5': (DIR_BOTH, EXP_LOWER),
    #'bill_amt6': (DIR_BOTH, EXP_LOWER),
    #'pay_amt1': (DIR_BOTH, EXP_HIGHER),
    #'pay_amt2': (DIR_BOTH, EXP_HIGHER),
    #'pay_amt3': (DIR_BOTH, EXP_HIGHER),
    #'pay_amt4': (DIR_BOTH, EXP_HIGHER),
    #'pay_amt5': (DIR_BOTH, EXP_HIGHER),
    #'pay_amt6': (DIR_BOTH, EXP_HIGHER)
}

def get_credit_default_cost_funcs(feature_info, data, sens_group=None):
    """Returns fractional cost functions taking into account the semantics of the Credit Card Default dataset"""
    return get_dataset_cost_funcs(CREDIT_DEFAULT_DIRS, feature_info, data, sens_group)

PRE_PROC_CREDIT_DEFAULT_DIRS = {
    'Male': (DIR_IMMUT, None),
    'Married': (DIR_IMMUT, None),
    'Single': (DIR_IMMUT, None),
    'Age_lt_25': (DIR_IMMUT, None),
    'Age_in_25_to_40': (DIR_IMMUT, None),
    'Age_in_40_to_59': (DIR_IMMUT, None),
    'Age_geq_60': (DIR_IMMUT, None),
    'EducationLevel': (DIR_UP, EXP_HIGHER),
    'MaxBillAmountOverLast6Months': (DIR_BOTH, EXP_HIGHER),
    'MaxPaymentAmountOverLast6Months': (DIR_BOTH, EXP_HIGHER),
    'MonthsWithZeroBalanceOverLast6Months': (DIR_BOTH, EXP_LOWER),
    'MonthsWithLowSpendingOverLast6Months': (DIR_BOTH, EXP_LOWER),
    'MonthsWithHighSpendingOverLast6Months': (DIR_BOTH, EXP_HIGHER),
    'MostRecentBillAmount': (DIR_BOTH, EXP_HIGHER),
    'MostRecentPaymentAmount': (DIR_BOTH, EXP_HIGHER),
    'TotalOverdueCounts': (DIR_BOTH, EXP_LOWER),
    'TotalMonthsOverdue': (DIR_BOTH, EXP_LOWER),
    'HistoryOfOverduePayments': (DIR_BOTH, EXP_LOWER),
}

PRE_PROC_CREDIT_DEFAULT_DIRS_REV = {
    'Male': (DIR_IMMUT, None),
    'Married': (DIR_IMMUT, None),
    'Single': (DIR_IMMUT, None),
    'Age_lt_25': (DIR_IMMUT, None),
    'Age_in_25_to_40': (DIR_IMMUT, None),
    'Age_in_40_to_59': (DIR_IMMUT, None),
    'Age_geq_60': (DIR_IMMUT, None),
    'EducationLevel': (DIR_UP, EXP_HIGHER),
    'MaxBillAmountOverLast6Months': (DIR_BOTH, EXP_LOWER),
    'MaxPaymentAmountOverLast6Months': (DIR_BOTH, EXP_LOWER),
    'MonthsWithZeroBalanceOverLast6Months': (DIR_BOTH, EXP_HIGHER),
    'MonthsWithLowSpendingOverLast6Months': (DIR_BOTH, EXP_HIGHER),
    'MonthsWithHighSpendingOverLast6Months': (DIR_BOTH, EXP_LOWER),
    'MostRecentBillAmount': (DIR_BOTH, EXP_LOWER),
    'MostRecentPaymentAmount': (DIR_BOTH, EXP_LOWER),
    'TotalOverdueCounts': (DIR_BOTH, EXP_HIGHER),
    'TotalMonthsOverdue': (DIR_BOTH, EXP_HIGHER),
    'HistoryOfOverduePayments': (DIR_BOTH, EXP_HIGHER),
}

def get_pre_proc_credit_default_cost_funcs(feature_info, data, sens_group=None, directions=PRE_PROC_CREDIT_DEFAULT_DIRS):
    """Returns fractional cost functions taking into account the semantics of the Credit Card Default dataset"""
    return get_dataset_cost_funcs(directions, feature_info, data, sens_group)


STUDENT_PERF_DIRS = {
    'school': (DIR_BOTH, EXP_LOWER),
    'sex': (DIR_IMMUT, None),
    'age': (DIR_IMMUT, None),
    'address': (DIR_BOTH, EXP_HIGHER),
    'famsize': (DIR_IMMUT, None),
    'Pstatus': (DIR_IMMUT, None),
    'Medu': (DIR_UP, EXP_HIGHER),
    'Fedu': (DIR_UP, EXP_HIGHER),
    'Mjob_at_home': (DIR_IMMUT, None),
    'Mjob_health': (DIR_IMMUT, None),
    'Mjob_other': (DIR_IMMUT, None),
    'Mjob_services': (DIR_IMMUT, None),
    'Mjob_teacher': (DIR_IMMUT, None),
    'Fjob_at_home': (DIR_IMMUT, None),
    'Fjob_health': (DIR_IMMUT, None),
    'Fjob_other': (DIR_IMMUT, None),
    'Fjob_services': (DIR_IMMUT, None),
    'Fjob_teacher': (DIR_IMMUT, None),
    'reason_course': (DIR_IMMUT, None),
    'reason_home': (DIR_IMMUT, None),
    'reason_other': (DIR_IMMUT, None),
    'reason_reputation': (DIR_IMMUT, None),
    'guardian_father': (DIR_IMMUT, None),
    'guardian_mother': (DIR_IMMUT, None),
    'guardian_other': (DIR_IMMUT, None),
    'traveltime': (DIR_BOTH, EXP_HIGHER),
    'studytime': (DIR_BOTH, EXP_HIGHER),
    'failures': (DIR_IMMUT, None), # depends on past, can't change it
    'schoolsup': (DIR_BOTH, EXP_HIGHER),
    'famsup': (DIR_BOTH, EXP_HIGHER),
    'paid': (DIR_BOTH, EXP_HIGHER),
    'activities': (DIR_BOTH, EXP_HIGHER),
    'nursery': (DIR_IMMUT, None), # depends on history, can't change it
    'higher': (DIR_BOTH, EXP_HIGHER),
    'internet': (DIR_BOTH, EXP_HIGHER),
    'romantic': (DIR_BOTH, EXP_HIGHER),
    'famrel': (DIR_BOTH, EXP_HIGHER),
    'freetime': (DIR_BOTH, EXP_HIGHER),
    'goout': (DIR_BOTH, EXP_HIGHER),
    'Dalc': (DIR_BOTH, EXP_HIGHER),
    'Walc': (DIR_BOTH, EXP_HIGHER),
    'health': (DIR_BOTH, EXP_HIGHER),
    'absences': (DIR_BOTH, EXP_HIGHER),
    'G1': (DIR_BOTH, EXP_HIGHER),
    'G2': (DIR_BOTH, EXP_HIGHER),
}

STUDENT_PERF_DIRS_REV = {
    'school': (DIR_BOTH, EXP_LOWER),
    'sex': (DIR_IMMUT, None),
    'age': (DIR_IMMUT, None),
    'address': (DIR_BOTH, EXP_LOWER),
    'famsize': (DIR_IMMUT, None),
    'Pstatus': (DIR_IMMUT, None),
    'Medu': (DIR_UP, EXP_HIGHER),
    'Fedu': (DIR_UP, EXP_HIGHER),
    'Mjob_at_home': (DIR_IMMUT, None),
    'Mjob_health': (DIR_IMMUT, None),
    'Mjob_other': (DIR_IMMUT, None),
    'Mjob_services': (DIR_IMMUT, None),
    'Mjob_teacher': (DIR_IMMUT, None),
    'Fjob_at_home': (DIR_IMMUT, None),
    'Fjob_health': (DIR_IMMUT, None),
    'Fjob_other': (DIR_IMMUT, None),
    'Fjob_services': (DIR_IMMUT, None),
    'Fjob_teacher': (DIR_IMMUT, None),
    'reason_course': (DIR_IMMUT, None),
    'reason_home': (DIR_IMMUT, None),
    'reason_other': (DIR_IMMUT, None),
    'reason_reputation': (DIR_IMMUT, None),
    'guardian_father': (DIR_IMMUT, None),
    'guardian_mother': (DIR_IMMUT, None),
    'guardian_other': (DIR_IMMUT, None),
    'traveltime': (DIR_BOTH, EXP_LOWER),
    'studytime': (DIR_BOTH, EXP_LOWER),
    'failures': (DIR_IMMUT, None), # depends on past, can't change it
    'schoolsup': (DIR_BOTH, EXP_LOWER),
    'famsup': (DIR_BOTH, EXP_LOWER),
    'paid': (DIR_BOTH, EXP_LOWER),
    'activities': (DIR_BOTH, EXP_LOWER),
    'nursery': (DIR_IMMUT, None), # depends on history, can't change it
    'higher': (DIR_BOTH, EXP_LOWER),
    'internet': (DIR_BOTH, EXP_LOWER),
    'romantic': (DIR_BOTH, EXP_LOWER),
    'famrel': (DIR_BOTH, EXP_LOWER),
    'freetime': (DIR_BOTH, EXP_LOWER),
    'goout': (DIR_BOTH, EXP_LOWER),
    'Dalc': (DIR_BOTH, EXP_LOWER),
    'Walc': (DIR_BOTH, EXP_LOWER),
    'health': (DIR_BOTH, EXP_LOWER),
    'absences': (DIR_BOTH, EXP_LOWER),
    'G1': (DIR_BOTH, EXP_LOWER),
    'G2': (DIR_BOTH, EXP_LOWER),
}

def get_student_perf_cost_funcs(feature_info, data, sens_group=None, directions=STUDENT_PERF_DIRS):
    """Returns fractional cost functions taking into account the semantics of the Student Performance dataset"""
    return get_dataset_cost_funcs(directions, feature_info, data, sens_group)

CRIMES_DIRS = {
    'population': (DIR_BOTH, EXP_HIGHER),
    'householdsize': (DIR_BOTH, EXP_HIGHER), 
    'racepctblack': (DIR_IMMUT, None), 
    'racePctWhite': (DIR_IMMUT, None), 
    'racePctAsian': (DIR_IMMUT, None),
    'racePctHisp': (DIR_IMMUT, None),
    'agePct12t21': (DIR_BOTH, EXP_HIGHER),
    'agePct12t29': (DIR_BOTH, EXP_HIGHER),
    'agePct16t24': (DIR_BOTH, EXP_HIGHER),
    'agePct65up': (DIR_BOTH, EXP_HIGHER),
    'numbUrban': (DIR_BOTH, EXP_HIGHER),
    'pctUrban': (DIR_BOTH, EXP_HIGHER),
    'medIncome': (DIR_BOTH, EXP_HIGHER),
    'pctWWage': (DIR_BOTH, EXP_HIGHER),
    'pctWFarmSelf': (DIR_BOTH, EXP_HIGHER),
    'pctWInvInc': (DIR_BOTH, EXP_HIGHER),
    'pctWSocSec': (DIR_BOTH, EXP_HIGHER),
    'pctWPubAsst': (DIR_BOTH, EXP_HIGHER),
    'pctWRetire': (DIR_BOTH, EXP_HIGHER),
    'medFamInc': (DIR_BOTH, EXP_HIGHER),
    'perCapInc': (DIR_BOTH, EXP_HIGHER)
}

CRIMES_DIRS_REV = {
    'population': (DIR_BOTH, EXP_LOWER),
    'householdsize': (DIR_BOTH, EXP_LOWER), 
    'racepctblack': (DIR_IMMUT, None), 
    'racePctWhite': (DIR_IMMUT, None), 
    'racePctAsian': (DIR_IMMUT, None),
    'racePctHisp': (DIR_IMMUT, None),
    'agePct12t21': (DIR_BOTH, EXP_LOWER),
    'agePct12t29': (DIR_BOTH, EXP_LOWER),
    'agePct16t24': (DIR_BOTH, EXP_LOWER),
    'agePct65up': (DIR_BOTH, EXP_LOWER),
    'numbUrban': (DIR_BOTH, EXP_LOWER),
    'pctUrban': (DIR_BOTH, EXP_LOWER),
    'medIncome': (DIR_BOTH, EXP_LOWER),
    'pctWWage': (DIR_BOTH, EXP_LOWER),
    'pctWFarmSelf': (DIR_BOTH, EXP_LOWER),
    'pctWInvInc': (DIR_BOTH, EXP_LOWER),
    'pctWSocSec': (DIR_BOTH, EXP_LOWER),
    'pctWPubAsst': (DIR_BOTH, EXP_LOWER),
    'pctWRetire': (DIR_BOTH, EXP_LOWER),
    'medFamInc': (DIR_BOTH, EXP_LOWER),
    'perCapInc': (DIR_BOTH, EXP_LOWER)
}

def get_crimes_and_communities_cost_funcs(feature_info, data, sens_group=None, directions=CRIMES_DIRS):
    """Returns fractional cost functions taking into account the semantics of the Student Performance dataset"""
    return get_dataset_cost_funcs(directions, feature_info, data, sens_group)
