
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances


# ## Load Data

# In[2]:

df = pd.read_csv('student-por.csv', sep=';')
print (df.columns)
df.head()


# In[3]:

df.head()


# ## Convert binary feature values to 1 and 0

# In[4]:

# 1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira) 
# 2 sex - student's sex (binary: 'F' - female or 'M' - male) 
# 3 age - student's age (numeric: from 15 to 22) 
# 4 address - student's home address type (binary: 'U' - urban or 'R' - rural) 
# 5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3) 
# 6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart) 
# 7 Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education) 
# 8 Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education) 
# 9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
# 10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
# 11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other') 
# 12 guardian - student's guardian (nominal: 'mother', 'father' or 'other') 
# 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour) 
# 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours) 
# 15 failures - number of past class failures (numeric: n if 1<=n<3, else 4) 
# 16 schoolsup - extra educational support (binary: yes or no) 
# 17 famsup - family educational support (binary: yes or no) 
# 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no) 
# 19 activities - extra-curricular activities (binary: yes or no) 
# 20 nursery - attended nursery school (binary: yes or no) 
# 21 higher - wants to take higher education (binary: yes or no) 
# 22 internet - Internet access at home (binary: yes or no) 
# 23 romantic - with a romantic relationship (binary: yes or no) 
# 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent) 
# 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high) 
# 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high) 
# 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high) 
# 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high) 
# 29 health - current health status (numeric: from 1 - very bad to 5 - very good) 
# 30 absences - number of school absences (numeric: from 0 to 93) 

# # these grades are related with the course subject, Math or Portuguese: 
# 31 G1 - first period grade (numeric: from 0 to 20) 
# 31 G2 - second period grade (numeric: from 0 to 20) 
# 32 G3 - final grade (numeric: from 0 to 20, output target)

df['school'][df['school'] == 'GP'] = 1 # GP is 1, MS is 0
df['school'][df['school'] == 'MS'] = 0
df['sex'][df['sex'] == 'F'] = 1 # Female is 1, Male is 0
df['sex'][df['sex'] == 'M'] = 0
df['address'][df['address'] == 'U'] = 1 # U is 1, R is 0
df['address'][df['address'] == 'R'] = 0
df['famsize'][df['famsize'] == 'LE3'] = 0 # LE3 is 0, GT3 is 1
df['famsize'][df['famsize'] == 'GT3'] = 1
df['Pstatus'][df['Pstatus'] == 'T'] = 1 # T is 1, A is 0
df['Pstatus'][df['Pstatus'] == 'A'] = 0
# df[df['Mjob'] == 'teacher'] = 1 # categorical values
# df[df['Mjob'] == 'health'] = 2
# df[df['Mjob'] == 'services'] = 3
# df[df['Mjob'] == 'home'] = 4
# df[df['Mjob'] == 'other'] = 5
# df[df['Fjob'] == 'teacher'] = 1 # same as MJob
# df[df['Fjob'] == 'health'] = 2
# df[df['Fjob'] == 'services'] = 3
# df[df['Fjob'] == 'home'] = 4
# df[df['Fjob'] == 'other'] = 5
df['schoolsup'][df['schoolsup'] == 'yes'] = 1
df['schoolsup'][df['schoolsup'] == 'no'] = 0
df['famsup'][df['famsup'] == 'yes'] = 1
df['famsup'][df['famsup'] == 'no'] = 0
df['paid'][df['paid'] == 'yes'] = 1
df['paid'][df['paid'] == 'no'] = 0
df['activities'][df['activities'] == 'yes'] = 1
df['activities'][df['activities'] == 'no'] = 0
df['nursery'][df['nursery'] == 'yes'] = 1
df['nursery'][df['nursery'] == 'no'] = 0
df['higher'][df['higher'] == 'yes'] = 1
df['higher'][df['higher'] == 'no'] = 0
df['internet'][df['internet'] == 'yes'] = 1
df['internet'][df['internet'] == 'no'] = 0
df['romantic'][df['romantic'] == 'yes'] = 1
df['romantic'][df['romantic'] == 'no'] = 0


# ## Drop all features except gender that are immutable

# In[5]:

cols_to_drop = ["Mjob", 
"Fjob", 
"reason", 
"guardian", 
#"schoolsup", 
#"famsup", 
# "paid",
# "activities",
"nursery",
# "higher",
# "internet",
# "romantic",
# "famrel",
# "freetime",
# "goout",
# "Dalc",
# "Walc",
# "health",
# "absences",
"Pstatus",
"famsize", 
"age", 
"failures"
]
df.drop(columns=cols_to_drop, inplace=True)


# ## Check if all rows are unique based on the new set of features

# In[6]:

def same_cols(v1, v2):
    if np.all(v1 == v2):
        return 1
    else:
        return 0
def check_all_rows_unique(df):
    if np.all(np.sum(pairwise_distances(np.array(df), metric=same_cols), axis=1) == 1):
        return True
    else:
        return False, "{} entries > 1".format(np.count_nonzero(np.sum(pairwise_distances(np.array(df), metric=same_cols), axis=1) > 1))
check_all_rows_unique(df)


# In[7]:

df.head()


# ## Create a list containing the name of each feature along with it's type (eg: continuous or binary)

# In[8]:

feature_info = []
X = None
for fname in df.columns:
    if fname in ['Mjob', 'Fjob', 'reason', 'guardian']:
        ohe = OneHotEncoder()
        new_df = ohe.fit_transform(np.array([df[fname]]).reshape((len(df), 1))).toarray()
        X = new_df if X is None else np.append(X, new_df, axis=1)
        for cat in ohe.categories_[0]:
            feature_info.append(('{}_{}'.format(fname, cat), 'cat_bin', ['0']))
        print (fname, new_df, ohe.categories_)
        print ()
    elif fname in ['school','sex','address','famsize','Pstatus',
                   'schoolsup','famsup','paid','activities',
                   'nursery','higher','internet','romantic']:
        X = np.array(df[fname]).reshape((len(df), 1)) if X is None else             np.append(X, np.array(df[fname]).reshape((len(df), 1)), axis=1)
        print (X.shape, np.array(df[fname]).reshape((len(df), 1)).shape)
        feature_info.append(('{}'.format(fname), 'cat_bin', ['0']))
    elif fname == 'G3':
        Y = np.array(df[fname]).flatten()
    else:
        X = np.array(df[fname]).reshape((len(df), 1)) if X is None else             np.append(X, np.array(df[fname]).reshape((len(df), 1)), axis=1)
        feature_info.append(('{}'.format(fname), 'cont', ['<num>']))


# In[9]:

X.shape, X, Y.shape, Y


# ## Construct x_control, indicating the name of the sensitive feature; here we choose sex_Male as the sensitive feature. This key corresponds to a numpy array indicating whether the subject is Male (True) or not (False)

# In[10]:

feature_names = []
for f in feature_info:
    print ("'{}':".format(f[0]))
    feature_names.append(f[0])
feature_info
x_control = {'sex_Male':np.array(df['sex'] == 0).flatten()}
x_control


# ## Create a dictionary holding the dataset together and dump it as a pickle file
# 
# #### X: is a matrix with each row corresponding to a user and Y is a vector indicating ground truth of each user, attr_info is a list of tuples where each tuple is 3 dimensional (ft_name, ft_type, [ft_values]), attr_names is a list of all atribute names and x_control contains information about the sensitive group

# In[11]:

def make_dataset(X, Y, feature_info, feature_names, x_control):
    ds = {}
    ds['X'] = X.astype('float')
    ds['Y'] = Y.astype('float')
    ds['attr_info'] = feature_info
    ds['attr_names'] = feature_names
    ds['x_control'] = x_control
    return ds

ds = make_dataset(X, Y, feature_info, feature_names, x_control)
joblib.dump(ds, 'processed_student_por.pkl')


# In[12]:

# sanity check
ds = joblib.load('processed_student_por.pkl')
assert np.all(ds['X'] == X) and np.all(ds['Y'] == Y)


# In[13]:

with open('processed_student_data.csv','w') as fp:
    df.to_csv(fp, index=False)


# In[ ]:



