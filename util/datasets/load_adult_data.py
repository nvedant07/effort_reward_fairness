import re
from . import file_util as fu

"""
    The adult dataset can be obtained from: http://archive.ics.uci.edu/ml/datasets/Adult
    The code will look for the data files (adult.data, adult.test) in the present directory, if they are not found, it will download them from UCI archive.
"""

def load_adult_data(load_data_size=None, normalize_cont_features=True, filter_minority_sens_groups=False):

    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'] # all attributes
    cont_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'] # attributes with integer values -- the rest are categorical
    attrs_to_ignore = {'sex', 'race' ,'fnlwgt'} # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    #sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
    sensitive_attrs = {'race', 'sex'} # the fairness constraints will be used for this feature
    attrs_to_ignore -= sensitive_attrs
    #rel_sens_vals = ['White', 'Black', 'Other', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander']
    if filter_minority_sens_groups:
        rel_sens_vals = ['race_White', 'race_Black', 'race_Asian-Pac-Islander']
    else:
        rel_sens_vals = None
    #no_hot_encode_attrs = {'native_country'} # the way we encode native country, its binary so no need to apply one hot encoding on it

    filtered_attrs = [attr for attr in attrs if attr not in attrs_to_ignore]
    attr_map = {'attrs': filtered_attrs, 'cont_attrs': cont_attrs,
            'sens_attrs': sensitive_attrs, 'rel_sens_vals': rel_sens_vals}

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    DATASET_FOLDER = 'adult'
    data_files = ["adult.data", "adult.test"]
    ADULT_WEB_RESOURCE = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    data_generator = fu.get_data_rows(DATASET_FOLDER,
            ADULT_WEB_RESOURCE, data_files)

    def filter_data(data_input):
        for line in data_input:
            if len(line) == len(attrs) + 1 and "?" not in line: # if a line has missing attributes, ignore it
                yield line
    data_generator = filter_data(data_generator)

    def convert_data(data_input):
        for line in data_input:
            for i, (attr_val, attr_name) in enumerate(zip(line, attrs)):
                # reducing dimensionality of some very sparse features
                if attr_name == "native_country":
                    if attr_val!="United-States":
                        attr_val = "Non-United-States"
                elif attr_name == "education":
                    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th", "10th", "11th", "12th"]:
                        attr_val = "high-school"
                line[i] = attr_val
            yield line
    data_generator = convert_data(data_generator)

    def filter_features_label(data_input):
        for line in data_input:
            assert len(line) == len(attrs) + 1

            class_label = line[-1]
            if class_label in ["<=50K.", "<=50K"]:
                class_label = -1
            elif class_label in [">50K.", ">50K"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")

            # filter features
            line = [attr for attr, attr_name in zip(line, attrs) if attr_name not in attrs_to_ignore]
            yield line, class_label
    data_generator = filter_features_label(data_generator)

    return fu.load_data(data_generator, attr_map, load_data_size, normalize_cont_features)


def load_census_income_data(load_data_size=None):

    attrs = ["age", "class of worker", "detailed industry recode", "detailed occupation recode", "education", "wage per hour", "enroll in edu inst last wk", "marital stat", "major industry code", "major occupation code", "race", "hispanic origin", "sex", "member of a labor union", "reason for unemployment", "full or part time employment stat", "capital gains", "capital losses", "dividends from stocks", "tax filer stat", "region of previous residence", "state of previous residence", "detailed household and family stat", "detailed household summary in household", "instance weight", "migration code-change in msa", "migration code-change in reg", "migration code-move within reg", "live in this house 1 year ago", "migration prev res in sunbelt", "num persons worked for employer", "family members under 18", "country of birth father", "country of birth mother", "country of birth self", "citizenship", "own business or self employed", "fill inc questionnaire for veteran's admin", "veterans benefits", "weeks worked in year", "year"]

    cont_attrs = ['age', 'capital gains', 'capital losses', 'dividends from stocks', 'num persons worked for employer', 'weeks worked in year'] # attributes with integer values -- the rest are categorical
    attrs_to_keep = {"age", "class of worker", "education", "marital stat", "major industry code", "major occupation code", "capital gains", "capital losses", "dividends from stocks", "num persons worked for employer", "country of birth self", "own business or self employed", "weeks worked in year"}
    #sensitive_attrs = {'race'} # the fairness constraints will be used for this feature
    sensitive_attrs = set()
    attrs_to_keep |= sensitive_attrs
    #rel_sens_vals = ['White', 'Black', 'Other', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander']
    rel_sens_vals = ['race_White', 'race_Black', 'race_Asian-Pac-Islander']
    #rel_sens_vals = set()
    #no_hot_encode_attrs = {'country of birth self'} # the way we encode native country, its binary so no need to apply one hot encoding on it

    filtered_attrs = [attr for attr in attrs if attr in attrs_to_keep]
    attr_map = {'attrs': filtered_attrs, 'cont_attrs': cont_attrs,
            'sens_attrs': sensitive_attrs, 'rel_sens_vals': rel_sens_vals}
            

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    DATASET_FOLDER = 'census_income'
    data_files = ["census-income.data.gz", "census-income.test.gz"]
    CENSUS_WEB_RESOURCE = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/"
    data_generator = fu.get_data_rows(DATASET_FOLDER, CENSUS_WEB_RESOURCE, data_files)

    def filter_data(data_input):
        for line in data_input:
            if len(line) != len(attrs) + 1: # if a line has missing attributes, ignore it
                continue
            for attr_val, attr_name in zip(line, attrs):
                if attr_name in attrs_to_keep \
                        and attr_val == 'Not in universe':
                    break
                # filters as for adult data
                if attr_name == 'age' and int(attr_val) <= 17:
                    break
                if attr_name == 'instance weight' and float(attr_val) < 1:
                    break
                if attr_name == 'weeks worked in year' \
                        and int(attr_val) < 1:
                    break
                if attr_name == 'wage per hour' \
                        and int(attr_val) == 0:
                    break
            else:
                yield line
    data_generator = filter_data(data_generator)

    def convert_data(data_input):
        for line in data_input:
            for i, (attr_val, attr_name) in enumerate(zip(line, attrs)):
                # reducing dimensionality of some very sparse features
                if attr_name == "country of birth self":
                    if attr_val!="United-States":
                        attr_val = "Non-United-Stated"
                elif attr_name == "education":
                    if attr_val in ["Children", "Less than 1st grade",
                            "1st 2nd 3rd or 4th grade",
                            "5th or 6th grade", "7th and 8th grade"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th grade", "10th grade",
                            "11th grade", "12th grade no diploma"]:
                        attr_val = "high-school"
                    else:
                        attr_val = "higher-edu"
                line[i] = attr_val
            yield line

    def filter_features_label(data_input):
        for line in data_input:

            assert len(line) == len(attrs) + 1

            class_label = float(line[attrs.index('wage per hour')]) / 100.
            #print('wage:', class_label)

            # filter features
            line = [attr for attr, attr_name in zip(line, attrs) if attr_name in attrs_to_keep]
            yield line, class_label
    data_generator = filter_features_label(data_generator)

    data = fu.load_data(data_generator, attr_map, load_data_size)
    return data

def load_communities_crime_data(load_data_size=None):

    DATASET_FOLDER = 'communities_crime'
    attrs = []
    attrs_to_ignore = set()
    attrs_file = fu.check_data_file(DATASET_FOLDER, None,
            'communities_crime.attrs')
    with open(attrs_file, 'r') as attr_file:
        for line in attr_file:
            line = line.strip()
            if line == '':
                continue
            line = line.split(' ')
            assert len(line) == 2
            if line[0] == 'i': # attr to ignore
                attrs_to_ignore.add(line[1])
            else:
                attrs.append(line[0])

    sensitive_attrs = set()
    attrs_to_ignore -= sensitive_attrs
    rel_sens_vals = []

    filtered_attrs = [attr for attr in attrs if attr not in attrs_to_ignore]
    attr_map = {'attrs': filtered_attrs, 'cont_attrs': filtered_attrs,
            'sens_attrs': sensitive_attrs, 'rel_sens_vals': rel_sens_vals}

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    data_files = ['communities.data']
    COMMUNITIES_CRIMIE_WEB_RESOURCE = \
            'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/'
    data_generator = fu.get_data_rows(DATASET_FOLDER,
            COMMUNITIES_CRIMIE_WEB_RESOURCE, data_files)

    filtered_positions = [i for i, attr in enumerate(attrs) if attr not in attrs_to_ignore]
    def filter_data(data_input):
        for line in data_input:
            if len(line) != len(attrs) + 1: # if a line has missing attributes, ignore it
                continue
            if any(line[i] == '?' for i in filtered_positions):
                continue
            yield line
    data_generator = filter_data(data_generator)

    def filter_features_label(data_input):
        for line in data_input:
            assert len(line) == len(attrs) + 1

            class_label = float(line[-1])

            # filter features
            line = [attr for attr, attr_name in zip(line, attrs) if attr_name not in attrs_to_ignore]
            yield line, class_label
    data_generator = filter_features_label(data_generator)

    return fu.load_data(data_generator, attr_map, load_data_size)

