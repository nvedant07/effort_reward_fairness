import os
import gzip
import numpy as np
from random import seed, shuffle
import urllib.request, urllib.error, urllib.parse
from sklearn import preprocessing
import xlrd
import sys
sys.path.insert(0, '../')
import output

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

ATTR_CAT_BIN = 'cat_bin'
ATTR_CAT_MULT = 'cat_mult'
ATTR_CONT = 'cont'
ATTR_CONT_QUANT = 'cont_quant'

REL_SENS_VALS_KEY = 'rel_sens_vals'

def load_data(data_input, attr_map, load_data_size=None, normalize_cont_features=True):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = attr_map['attrs']
    cont_attrs = attr_map['cont_attrs']
    sensitive_attrs = attr_map['sens_attrs']

    rel_sens_vals = attr_map[REL_SENS_VALS_KEY] if REL_SENS_VALS_KEY in attr_map else {}


    attrs_to_vals = {k: list() for k in attrs} # will store the values for each attribute for all users
    y = []

    for line, class_label in data_input:
        y.append(class_label)

        for attr_val, attr_name in zip(line, attrs):
            attrs_to_vals[attr_name].append(attr_val)

    # attrs_to_vals is a dict of type {attr_name : vector of values of that attribute (a column in the matrix X)}

    X = []
    x_control = {}
    #attr_names = []
    attr_infos = []

    # if the integer vals are not binary, we need to get one-hot encoding for them
    for i, attr_name in enumerate(attrs):
        attr_vals = attrs_to_vals[attr_name]

        attr_val_labels = []

        if attr_name in cont_attrs:
            # Do not touch continuous attributes; they will be one-hot encoded later based on ripper or whatever training rule is used
            new_vals = [preprocessing.scale(attr_vals)] if normalize_cont_features else [attr_vals]
            #new_names = [attr_name]
            attr_type = ATTR_CONT
            attr_val_labels.append('<num>')
        else:
            lb = preprocessing.LabelBinarizer()
            attr_vals = lb.fit_transform(attr_vals)
            #print ("attr vals:", attr_vals)
            new_vals = []
            #new_names = []
            if len(lb.classes_) > 2:
                attr_type = ATTR_CAT_MULT
                for j, (label, inner_col) in enumerate(zip(lb.classes_, attr_vals.T)):
                    new_vals.append(inner_col)
                    attr_val_labels.append(label)
            else:
                # binary feature
                attr_type = ATTR_CAT_BIN
                # pick the class with the shorter name
                if len(lb.classes_[0]) > len(lb.classes_[1]):
                    new_col = 1 - attr_vals.flatten()
                    new_val_label = lb.classes_[1]
                else:
                    new_col = attr_vals.flatten()
                    new_val_label = lb.classes_[0]
                new_vals.append(new_col)
                attr_val_labels.append(new_val_label)

        X.extend(new_vals)
        #attr_names.extend(new_names)
        attr_infos.append((attr_name, attr_type, attr_val_labels))

        # TODO: fix this and make it compatible with attr_info

        if attr_name in sensitive_attrs:

            for val_name, val_col in zip(attr_val_labels, new_vals):
                val_col = np.array(val_col, dtype=bool)
                full_val_name = attr_name + '_' + val_name
                if (rel_sens_vals and full_val_name in rel_sens_vals) or len(attr_val_labels) <= 2:
                    x_control[full_val_name] = val_col

    # convert to numpy arrays for easy handling
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype=float)
    #for k, v in list(x_control.items()): x_control[k] = np.array(v, dtype=float)

    # only keep people belonging to certain sensitive groups
    # if there are no senstive groups, select everyone
    idx = np.zeros(len(y), dtype=bool) if rel_sens_vals else \
            np.ones(len(y), dtype=bool)
    if rel_sens_vals:
        for k, v in x_control.items():
            if k in rel_sens_vals:
                print('{}: {} people'.format(k, sum(v)))
                idx = np.logical_or(idx, v)

    X = X[idx]
    y = y[idx]
    for k, v in x_control.items():
        x_control[k] = v[idx]
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in list(x_control.keys()):
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in list(x_control.keys()):
            x_control[k] = x_control[k][:load_data_size]

    print ('Loaded {} people, {} from pos and {} from neg class'.format(len(y),
            np.sum(y == 1.), np.sum(y == -1.)))
    print ('Attribute infos from load data: {}'.format(attr_infos))

    attr_names = attr_names_from_info(attr_infos)
    return {'X': X, 'Y': y, 'x_control': x_control,
            'attr_names': attr_names, 'attr_info': attr_infos}


def get_data_rows(dataset, base_addr, data_files, separator=',',
        inner_file_reader=None):
    for f in data_files:
        data_file = check_data_file(dataset, base_addr, f)

        print('Reading file', f)

        if f.endswith('.xls'):
            wb = xlrd.open_workbook(data_file)
            sheet = wb.sheet_by_index(0)
            for i in range(sheet.nrows):
                yield sheet.row_values(i)
        else:
            binary_input = f.endswith('.gz') or f.endswith('.zip')
            read_func = inner_file_reader if inner_file_reader is not None else gzip.open if binary_input else open
            for line in read_func(data_file, 'r'):
                if binary_input:
                    line = line.decode()
                line = line.strip()
                if line == "": continue # skip empty lines
                line = [el.strip() for el in line.split(separator)]
                yield line


def check_data_file(dataset, base_url, fname):
    #files = os.listdir(".") # get the current directory listing
    files_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/' + dataset # get path of this file
    output.create_dir(files_dir)
    files = os.listdir(files_dir) # get the current directory listing
    print ("Looking for file '%s' in the current directory..." % fname)
    full_file = "{}/{}".format(files_dir, fname)

    if fname not in files:
        print ("'{}' not found! Downloading ...".format(fname))
        url = base_url + urllib.parse.quote(fname)
        response = urllib.request.urlopen(url)
        content_charset = response.info().get_content_charset()
        if content_charset is not None:
            # string file
            data = response.read().decode(response.info().get_content_charset(), 'ignore')
            write_spec = "w"
        else:
            # binary file
            data = response.read()
            write_spec = "wb"
        with open(full_file, write_spec) as fileOut:
            fileOut.write(data)
        print("'%s' download and saved locally.." % fname)
    else:
        print("File found in current directory..")

    return full_file

def attr_names_from_info(attr_infos):
    attr_names = []
    for attr_name, attr_type, val_labels in attr_infos:
        if attr_type == ATTR_CONT:
            attr_names.append(attr_name)
        else:
            for val_label in val_labels:
                attr_names.append(attr_name + '_' + val_label)
    return attr_names
