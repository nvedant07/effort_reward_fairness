import numpy as np
import time
import sys
import pandas as pd

from . import learning_env
from . import eval_formula

sys.path.insert(0, '../util/')
from datasets import load_german_credit_data
from datasets import load_adult_data
from datasets import credit_card_default_data
from datasets import file_util
from datasets import load_student_perf_data
from datasets import load_crimes_and_communities

class DecRuleEnv(learning_env.LearningEnv):
    """Class to load and hold the data required for classification analysis"""

    def __init__(self, dataset, sens_group_desc, val_split=0.):
        #self.hyperparam_store_loc = 'hyperparams/prob_class_params'
        #output.create_dir('hyperparams')

        self.ds_name = dataset
        self.val_split = val_split
        self.sens_group_desc = sens_group_desc
        self.X = None

    def load_data(self, seed=4194, feature_engineering=False):
        """
        Load the dataset, shuffle the data and split the data into train, test
        and potentially validation
        """
        if self.X is None:
            if self.ds_name == 'Adult':
                data = load_adult_data.load_adult_data(normalize_cont_features=False)
                # TODO: why is this necessary for Ripper?
                data['Y'] = -data['Y']

            elif self.ds_name == 'GermanCredit':
                data = load_german_credit_data.load_german_credit_data(normalize_cont_features=False)

            elif self.ds_name == 'CreditDefault':
                data = credit_card_default_data.load_credit_card_default_data(normalize_cont_features=False)
                data['Y'] = -data['Y']

                if feature_engineering:
                    X, f_names, f_info = data['X'], data['attr_names'], data['attr_info']
                    X, f_names, f_info = process_credit_default_data(X, f_names, f_info)
                    #print("new fnames:", f_names)
                    #print("new finfo:", f_info)
                    data['X'] = X
                    data['attr_names'] = f_names
                    data['attr_info'] = f_info
            elif self.ds_name == 'PreprocCreditCardDefault':
                data = credit_card_default_data.load_preproc_credit_card_default_data(normalize_cont_features=False)
            elif self.ds_name == 'StudentPerf' or self.ds_name == "StudentPerfMutPlusImmut" or self.ds_name == "StudentPerfMut":
                data = load_student_perf_data.load_student_perf_data(self.ds_name)
            elif self.ds_name == 'CrimesCommunities':
                data = load_crimes_and_communities.load_crimes_and_communities()
            else:
                raise ValueError('Invalid dataset name "{}"'.format(self.ds_name))

            self.X = data['X']
            self.y = data['Y']
            self.feature_names = data['attr_names']
            self.feature_info = data['attr_info']
            self.x_control = data['x_control']

            if self.ds_name == 'PreprocCreditCardDefault':
                self.y = (self.y + 1.) / 2. # convert to {0, 1}

        np.random.seed(seed)
        order = np.random.permutation(len(self.X))
        self.X = self.X[order]
        self.y = self.y[order]
        self.x_control = {key: val[order] for key, val in self.x_control.items()}

        self.setup_data()

    def predict(self, X):
        return eval_formula.data_sat(self.model, X)

def process_credit_default_data(data, feature_names, feature_infos):
    pay_status, bill_amt, pay_amt = get_credit_default_indices(feature_names)
    """Feature engineering method for the Credit Card Default dataset"""
    last_pay_index = pay_status(5)
    assert feature_names[last_pay_index] == 'pay_6'
    # add demographic and payment status features
    cols = [data[:,i] for i in range(last_pay_index + 1)]
    new_feature_names = feature_names[:last_pay_index + 1]
    new_feature_infos = feature_infos[:11]

    pay_delay_months = np.sum((data[:,pay_status(0):pay_status(6)] > 0).astype(int), axis=1)
    cols.append(pay_delay_months)
    delay_name = "months_w_payment_delay"
    new_feature_names.append(delay_name)
    new_feature_infos.append((delay_name, file_util.ATTR_CONT, ['<num>']))

    for i_pay, j_bill in zip(range(5), range(1, 6)):
        pay_i_amt = data[:,pay_amt(i_pay)]
        bill_j_amt = data[:,bill_amt(j_bill)]

        # check that if bill_amt is 0, pay_amt should also be 0
        bill_zero = bill_j_amt == 0
        #assert np.any(pay_amt[bill_zero].astype(bool)) == False
        pay_i_amt[bill_zero] = 1.
        assert np.all(pay_i_amt >= 0)
        bill_j_amt[bill_zero] = 1.
        paid_frac_j = pay_i_amt / bill_j_amt
        # zero statement, so paid back fully
        paid_frac_j[paid_frac_j < 0] = 1.
        #print("faid_frac_j:", paid_frac_j[:10])
        #print("mean:", np.mean(paid_frac_j))

        cols.append(paid_frac_j / bill_j_amt)
        col_name = "frac_paid_{}".format(j_bill)
        new_feature_names.append(col_name)
        new_feature_infos.append((col_name, file_util.ATTR_CONT, ['<num>']))

    return np.column_stack(cols), new_feature_names, new_feature_infos

def get_credit_default_indices(feature_names):
    pay_status_offset = 12
    assert feature_names[pay_status_offset] == "pay_1"
    pay_status = lambda i: i + pay_status_offset
    bill_amt_offset = 18
    assert feature_names[bill_amt_offset] == "bill_amt1"
    bill_amt = lambda i: i + bill_amt_offset
    pay_amt_offset = 24
    assert feature_names[pay_amt_offset] == "pay_amt1"
    pay_amt = lambda i: i + pay_amt_offset

    return pay_status, bill_amt, pay_amt

def print_data(data, feature_names):
    print("shape data:", data.shape)
    print("num features:", len(feature_names))
    print("features:", feature_names)
    df = pd.DataFrame(data, columns=feature_names)
    print("Data:\n", df)
