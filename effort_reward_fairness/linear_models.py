from learning_env import learning_env
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
import scipy.io
import numpy as np

class LogReg(learning_env.LearningEnv):

    def __init__(self):
        self.x_val, self.y_val = None, None
        self.hyperparams = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'random_state': [42]}
        self.best_param = None

    def filename(self):
        return "logreg"

    def __str__(self):
        return "LogReg <<BR>> with reg_strength = {}".format(1./self.best_param['C']) if self.best_param is not None else "Logistic Regression"

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.best_param = self.find_hyperparams_cv(LogisticRegression, self.hyperparams)
        self.clf = LogisticRegression(**self.best_param)
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

class DT(learning_env.LearningEnv):

    def __init__(self):
        self.x_val, self.y_val = None, None
        self.hyperparams = {'max_depth': [1, 5, 10, 15, 20, 25, 30], 'random_state': [42]}
        self.best_param = None

    def filename(self):
        return "dtree"

    def __str__(self):
        return "DTree <<BR>> with max_depth = {}".format(self.best_param['max_depth']) if self.best_param is not None else "Decision Tree"

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.best_param = self.find_hyperparams_cv(DecisionTreeClassifier, self.hyperparams)
        self.clf = DecisionTreeClassifier(**self.best_param)
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class SVM(learning_env.LearningEnv):

    def __init__(self):
        self.x_val, self.y_val = None, None
        self.hyperparams = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'random_state': [42], 'kernel': ['rbf','linear']}
        self.best_param = None

    def filename(self):
        return "svm"

    def __str__(self):
        return "SVM <<BR>> with C = {} <<BR>> kernel = {}".format(self.best_param['C'], self.best_param['kernel']) if self.best_param is not None else "SVM"

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.best_param = self.find_hyperparams_cv(SVC, self.hyperparams)
        self.clf = SVC(**self.best_param)
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class FCLogReg(learning_env.LearningEnv):

    def __init__(self, proxy_type, cons_strength):
        import sys
        sys.path.append('../fairness_optimization/')
        from fair_models import FairnessConstrainedClassifier

        self.x_val, self.y_val = None, None
        self.proxy_type = proxy_type
        self.cons_strength = cons_strength
        self.sens_feature = 'Male_0'
        self.clf = FairnessConstrainedClassifier()

        cons_params = {'cons_type': 'cov',
                'proxy_type': proxy_type,
                'sens_feature': self.sens_feature}
        self.clf.add_constraint('cov_cons', cons_params)

    def filename(self):
        return "FcLogReg"

    def __str__(self):
        return "Fairness constrained LogisticRegression <<BR>> {} constraint, threshold = {}".format(
                self.proxy_type, self.cons_strength)

    def train(self, x_train, y_train, x_control):
        y_train = 2 * y_train - 1
        x_control = {k: v.astype(int) for k, v in x_control.items()}
        #self.x_train = x_train
        #self.y_train = y_train

        # Pretrain to compute constraint reference values
        # TODO: try to get rid of this step
        self.clf.fit(x_train, y_train, x_control, self.sens_feature)

        self.clf.set_cons_strength('cov_cons', self.cons_strength)
        self.clf.fit(x_train, y_train, x_control, self.sens_feature)

    def predict(self, x):
        return self.clf.predict(x)

class NN(learning_env.LearningEnv):
    """ A single layer neural network with ReLU on the hidden layer;
    solver is adam
    """
    def __init__(self):
        self.x_val, self.y_val = None, None
        self.hyperparams = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'random_state': [42], 
            'hidden_layer_sizes': [(50,), (60,), (70,), (80,), (90,), (100,)]}
        self.best_param = None

    def filename(self):
        return "neuralnet"

    def __str__(self):
        return "NeuralNet <<BR>> with hidden_layer_sizes = {} and <<BR>> regularization_strength = {}".format(
            self.best_param['hidden_layer_sizes'], self.best_param['alpha']) if self.best_param is not None else "NeuralNet"

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.best_param = self.find_hyperparams_cv(MLPClassifier, self.hyperparams)
        self.clf = MLPClassifier(**self.best_param)
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

class RidgeReg(learning_env.LearningEnv):
    """Linear Regression"""
    def __init__(self, alpha):
        self.x_val, self.y_val = None, None
        self.alpha = alpha

    def __str__(self):
        return "RidgeReg"

    def filename(self):
        return "RidgeReg"

    def shortfilename(self):
        return "Ridge"

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = Ridge(alpha=self.alpha)
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

class LinReg(learning_env.LearningEnv):
    """Linear Regression"""
    def __init__(self):
        self.x_val, self.y_val = None, None

    def __str__(self):
        return "LinearReg"

    def filename(self):
        return "LinearReg"

    def shortfilename(self):
        return "Linear"

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = LinearRegression(n_jobs=-1)
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

class NNReg(learning_env.LearningEnv):
    """Multi-layer perceptron for regression"""
    def __init__(self):
        self.x_val, self.y_val = None, None
        self.x_val, self.y_val = None, None
        self.hyperparams = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'random_state': [42], 
            'hidden_layer_sizes': [(50,), (60,), (70,), (80,), (90,), (100,)]}
        self.best_param = None

    def __str__(self):
        return "NeuralNet <<BR>> Regressor <<BR>> with hidden_layer_sizes = {} and <<BR>> regularization_strength = {}".format(
            self.best_param['hidden_layer_sizes'], self.best_param['alpha']) if self.best_param is not None else "NeuralNet Regressor"

    def filename(self):
        return "NeuralNetReg"

    def shortfilename(self):
        return "NN"

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.best_param = self.find_hyperparams_cv(MLPRegressor, self.hyperparams)
        self.clf = MLPRegressor(**self.best_param)
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

class DTReg(learning_env.LearningEnv):
    '''Regression Tree'''
    def __init__(self):
        self.x_val, self.y_val = None, None
        self.hyperparams = {'max_depth': [1, 5, 10, 15, 20, 25, 30], 'random_state': [42]}
        self.best_param = None

    def filename(self):
        return "DtreeReg"

    def shortfilename(self):
        return "DT"

    def __str__(self):
        return "DTreeReg <<BR>> with max_depth = {}".format(self.best_param['max_depth']) if self.best_param is not None else "Decision Tree"

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.best_param = self.find_hyperparams_cv(DecisionTreeRegressor, self.hyperparams)
        self.clf = DecisionTreeRegressor(**self.best_param)
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

class LinRegFC(learning_env.LearningEnv):
    '''Linear Regression with Fairness Constraints'''
    def __init__(self, tau, dataset):
        try:
            mat = scipy.io.loadmat('./trained_linregfc_{}.mat'.format(dataset))
        except:
            raise ValueError("trained_linregfc_{}.mat file not found".format(dataset))
        self.idx = np.where(mat['Tau'].flatten() == tau)[0][0]
        self.tau = tau
        self.W = mat['W_all'][:,self.idx]

    def filename(self):
        return "Tau_{}".format(self.tau)

    def shortfilename(self):
        return "LinearFC (tau = {})".format(self.tau)

    def __str__(self):
        return "FC ({}) <<BR>> with tau = {}".format(self.tau, self.tau)

    def train(self, x_train, y_train):
        pass

    def predict(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = np.append(x, 1) # append 1 for intercept
            prediction = np.dot(self.W, x)
        elif x.ndim == 2:
            x = np.append(x , np.array([1] * x.shape[0]).reshape(x.shape[0], 1), axis=1) # add a column of 1
            prediction = np.sum(np.vstack([self.W] * x.shape[0]) * x, axis=1)
        else:
            raise ValueError("Cannot have ndim > 2!")
        return prediction