import re
from . import file_util as fu

"""
    The German Credit dataset can be obtained from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    The code will look for the data file (german.data) in the present directory, if they are not found, it will download them from UCI archive.
"""

def load_german_credit_data(load_data_size=None, normalize_cont_features=True):

    attrs = ['checking_acc_status', 'duration', 'credit_hist', 'purpose', 'credit_amount', 'savings_acc', 'employment', 'installment_rate', 'sex', 'debtors_guarantors', 'residence_duration', 'property', 'age', 'installment_plans', 'housing', 'num_credit_cards', 'job', 'num_people_liable_to', 'telephone', 'foreign_worker'] # all attributes
    cont_attrs = ['duration', 'credit_amount', 'installment_rate', 'residence_duration', 'age', 'num_credit_cards', 'num_people_liable_to'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = {'sex', 'foreign_worker'}
    #rel_sens_vals = ['race_White', 'race_Black', 'race_Asian-Pac-Islander']

    attr_map = {'attrs': attrs, 'cont_attrs': cont_attrs,
            'sens_attrs': sensitive_attrs, 'rel_sens_vals': {}}

    DATASET_FOLDER = 'german'
    data_files = ["german.data"]
    GERMAN_WEB_RESOURCE = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
    data_generator = fu.get_data_rows(DATASET_FOLDER,
            GERMAN_WEB_RESOURCE, data_files, separator=' ')

    # converting the cryptic Axxx names to something more understandable
    informative_categories = {'checking_acc_status': {'A11': '<0DM' ,
                'A12': '[0,200)DM',
                'A13': '>=200DM',
                'A14': 'None'},
            'credit_hist': {'A30': 'none/all_paid_back_duly',
                'A31': 'all-paid-back-duly-at-bank',
                'A32': 'existing-paid-back-duly-till-now',
                'A33': 'delay-in-past',
                'A34': 'critical-acc/credits-at-other-bank'},
            'purpose': {'A40': 'new-car',
                'A41': 'used-car',
                'A42': 'furniture/equipment',
                'A43': 'radio/television',
                'A44': 'domestic-appliances',
                'A45': 'repairs', 'A46': 'education',
                'A47': 'vacation',
                'A48': 'retraining',
                'A49': 'business',
                'A410': 'others'},
            'savings_acc': {'A61': '<100DM',
                'A62': '[100,500)DM',
                'A63': '[500,1000)DM',
                'A64': '>=1000DM',
                'A65': 'unknown/none'},
            'employment': {'A71': 'none',
                'A72': '<1y',
                'A73': '[1,4)y',
                'A74': '[4,7)y',
                'A75': '>=7y'},
            'sex': {'A91': 'male',  # divorced/separated
                'A92': 'female',    #divorced/separated/married
                'A93': 'male',      #single
                'A94': 'male',      #married/widowed
                'A95': 'female'},   #single
            'debtors_guarantors': {'A101': 'none',
                'A102': 'co-applicant',
                'A103': 'guarantor'},
            'property': {'A121': 'real-estate',
                'A122': 'building-soc-saving-agreement/life-insurance', # if not A121
                'A123': 'car-or-other',                                 # not in A121,A122
                'A124': 'unknown/none'},
            'installment_plans': {'A141': 'bank',
                'A142': 'stores',
                'A143': 'none'},
            'housing': {'A151': 'rent',
                'A152': 'own',
                'A153': 'for-free'},
            'job': {'A171': 'unemployed/unskilled-non-resident',
                'A172': 'unskilled-resident',
                'A173': 'skilled-empl/official',
                'A174': 'management/self-employed/highly-qualified-empl/officer'},
            'telephone': {'A191': 'no', 'A192': 'yes'},
            'foreign_worker': {'A201': 'yes', 'A202': 'no'}}

    def convert_data(data_input):
        for line in data_input:
            assert len(line) == len(attrs) + 1
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
            if class_label == "1": # Good
                class_label = +1
            elif class_label == "2": # Bad
                class_label = -1
            else:
                raise Exception("Invalid class label value '{}'".format(class_label))

            yield line[:-1], class_label
    data_generator = filter_features_label(data_generator)

    return fu.load_data(data_generator, attr_map, load_data_size, normalize_cont_features)

