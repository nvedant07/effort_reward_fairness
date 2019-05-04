import re
from . import file_util as fu

DATASET_FOLDER = 'credit_card_default'

"""
    The default of credit card clients dataset can be obtained from: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    The code will look for the data file (default of credit card clients.xls) in the present directory, if they are not found, it will download them from UCI archive.
"""

def load_credit_card_default_data(load_data_size=None, normalize_cont_features=True):

    attrs = ['limit_bal', 'sex', 'education', 'marriage', 'age',
            'pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
            'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
            'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'] # + id as first feature
    cont_attrs = ['id', 'limit_bal',  'age',
		'pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
		'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
		'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']
    sensitive_attrs = {'sex'}
    #rel_sens_vals = ['race_White', 'race_Black', 'race_Asian-Pac-Islander']

    attr_map = {'attrs': attrs, 'cont_attrs': cont_attrs,
            'sens_attrs': sensitive_attrs}

    data_files = ['default of credit card clients.xls']
    CREDIT_DEFAULT_WEB_RESOURCE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/'
    data_generator = fu.get_data_rows(DATASET_FOLDER,
            CREDIT_DEFAULT_WEB_RESOURCE, data_files)

    # converting category numbers to understandable values
    informative_categories = {'sex': {1: 'male',
                2: 'female'},
            'education': {0: 'unknown',
                1: 'graduate-school',
                2: 'university',
                3: 'high-school',
                4: 'others',
                5: 'unknown',
                6: 'unknown'},
            'marriage': {0: 'unknown',
                1: 'married',
                2: 'single',
                3: 'others'}}

    def convert_data(data_input):
        for j, line in enumerate(data_input):
            # Skip first two header lines
            if j < 2:
                continue

            # + id + target label
            assert len(line) == len(attrs) + 2, "expected line length {}, but got {}".format(len(attrs) + 2, len(line))
            line = line[1:]
	
            for i, (attr_val, attr_name) in enumerate(zip(line, attrs)):
                if attr_name in informative_categories:
                    info_cat = informative_categories[attr_name]
                    line[i] = info_cat[attr_val]
            yield line
    data_generator = convert_data(data_generator)

    def filter_features_label(data_input):
        for line in data_input:
            assert len(line) == len(attrs) + 1

            class_label = line[-1]
            if class_label == 1: # Default
                class_label = +1
            elif class_label == 0: # No-Default
                class_label = -1
            else:
                raise Exception("Invalid class label value '{}'".format(class_label))

            yield line[:-1], class_label
    data_generator = filter_features_label(data_generator)

    return fu.load_data(data_generator, attr_map, load_data_size, normalize_cont_features)

def load_preproc_credit_card_default_data(load_data_size=None, normalize_cont_features=True):
    attrs = ["Male", "Married", "Single",
            "Age_lt_25", "Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60",
            "EducationLevel", "MaxBillAmountOverLast6Months", "MaxPaymentAmountOverLast6Months",
            "MonthsWithZeroBalanceOverLast6Months", "MonthsWithLowSpendingOverLast6Months",
            "MonthsWithHighSpendingOverLast6Months", "MostRecentBillAmount",
            "MostRecentPaymentAmount", "TotalOverdueCounts",
            "TotalMonthsOverdue", "HistoryOfOverduePayments"]
    cont_attrs = ["EducationLevel", "MaxBillAmountOverLast6Months", "MaxPaymentAmountOverLast6Months",
            "MonthsWithZeroBalanceOverLast6Months", "MonthsWithLowSpendingOverLast6Months",
            "MonthsWithHighSpendingOverLast6Months", "MostRecentBillAmount",
            "MostRecentPaymentAmount", "TotalOverdueCounts",
            "TotalMonthsOverdue", "HistoryOfOverduePayments"]
    sensitive_attrs = ['Male']

    attr_map = {'attrs': attrs, 'cont_attrs': cont_attrs,
            'sens_attrs': sensitive_attrs}

    data_files = ['credit_processed.csv']
    PROCESSED_CREDIT_DEFAULT_WEB_RESOURCE = 'https://raw.githubusercontent.com/ustunb/actionable-recourse/master/data/'
    data_generator = fu.get_data_rows(DATASET_FOLDER,
            PROCESSED_CREDIT_DEFAULT_WEB_RESOURCE, data_files)

    # skip the header
    header = next(data_generator)
    assert header[1:] == attrs

    def convert_features_label(data_input):
        for line in data_input:
            assert len(line) == len(attrs) + 1

            class_label = float(line[0])
            assert class_label in [0., 1.], "class label is {}".format(class_label)
            class_label = class_label * 2 - 1 # convert to -1, 1

            #yield [int(val) for val in line[1:]], class_label
            yield line[1:], class_label
    data_generator = convert_features_label(data_generator)

    return fu.load_data(data_generator, attr_map, load_data_size, normalize_cont_features)

