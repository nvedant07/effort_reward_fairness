import numpy as np
np.warnings.filterwarnings('ignore')
import cost_funcs as cf
from sklearn.metrics import pairwise_distances

def find_cost_of_y(y_i, y_j, y, sens_group):
    cost_i_s = np.count_nonzero(np.logical_and(y < y_i, sens_group))/np.count_nonzero(sens_group)
    cost_j_s = np.count_nonzero(np.logical_and(y < y_j, sens_group))/np.count_nonzero(sens_group)
    cost_i_ns = np.count_nonzero(np.logical_and(y < y_i, ~sens_group))/np.count_nonzero(~sens_group)
    cost_j_ns = np.count_nonzero(np.logical_and(y < y_j, ~sens_group))/np.count_nonzero(~sens_group)
    if y_i == y_j:
        assert cost_j_s == cost_i_s and cost_j_ns == cost_i_ns
    return max(0., max(cost_j_s - cost_i_s, cost_j_ns - cost_i_ns))

def dist(x_i, x_j, cost_funcs, cost_funcs_rev, sens_group, feature_info, users_mat, users_gt):
    ### Dirty but couldn't find any other way
    idx_i, idx_j = np.where(np.all(users_mat == x_i, axis=1))[0], np.where(np.all(users_mat == x_j, axis=1))[0]
    assert np.all(users_gt[idx_i] == users_gt[idx_i][0]) and np.all(users_gt[idx_j] == users_gt[idx_j][0]) and \
        np.all(sens_group[idx_i] == sens_group[idx_i][0]) and np.all(sens_group[idx_j] == sens_group[idx_j][0])
    idx_i, idx_j = idx_i[0], idx_j[0]
    d = find_cost_of_y(users_gt[idx_i], users_gt[idx_j], users_gt, sens_group)
    for k in range(len(feature_info)):
        fname, _, _ = feature_info[k]
        if not isinstance(cost_funcs[fname], cf.ImmutableCost):
            feature_cost_forward_s = cost_funcs[fname](x_i[k], x_j[k], True) # sens feature cost
            feature_cost_rev_s = cost_funcs_rev[fname](x_i[k], x_j[k], True)
            feature_cost_forward_ns = cost_funcs[fname](x_i[k], x_j[k], False) # non-sens feature cost
            feature_cost_rev_ns = cost_funcs_rev[fname](x_i[k], x_j[k], False)
            d += np.max((0, max(max(feature_cost_forward_s, feature_cost_rev_s), max(feature_cost_forward_ns, feature_cost_rev_ns) ) ) )
    if np.all(x_i == x_j):
        assert d == 0 and idx_i == idx_j
    return d

class Segregation:

    def __init__(self, sens_group, feature_info):
        self.sens_group = sens_group
        self.feature_info = feature_info

    def __str__(self):
        raise NotImplementedError("Please name me!")

    def shortname(self):
        raise NotImplementedError("Please give me a short name!")

    def val(self, **args):
        raise NotImplementedError("Implement how the index is calculated!")

class SSI(Segregation):

    def __init__(self, sens_group, feature_info):
        super(SSI, self).__init__(sens_group, feature_info)
        self.sens_group = self.sens_group[:10]
        self.n_s = np.count_nonzero(self.sens_group)

    def __str__(self):
        return "SSI"

    def shortname(self):
        return "ssi"

    def val(self, **args):
        B = pairwise_distances(X=args['X'][:10,:], metric=dist, n_jobs=-1, **{'cost_funcs': args['cost_funcs'], 'cost_funcs_rev': args['cost_funcs_rev'], 
            'sens_group' : self.sens_group[:10], 'feature_info': self.feature_info, 'users_mat': args['X'][:10,:], 'users_gt': args['y'][:10]})
        B = np.exp(-B)
        B = B/np.sum(B, axis=1)[:, None]
        print (B)
        # for i in range(B.shape[0]):

        w,v = np.linalg.eig(B)
        max_eigenval = np.max(w)
        max_eigenvector = v[:,np.argmax(w)]
        vector_sum = np.sum((max_eigenval * max_eigenvector).astype(float)[self.sens_group])
        determinant = np.linalg.det(B)
        ssi = vector_sum * determinant
        print ("SSI: {} ({} x {})".format(ssi, vector_sum, determinant))
        return ssi

class Atkinson(Segregation):

    def __init__(self, sens_group, feature_info):
        super(Atkinson, self).__init__(sens_group, feature_info)
        self.T = len(self.sens_group)
        self.P = np.count_nonzero(self.sens_group)/self.T
        self.beta = 0.5

    def __str__(self):
        return "AI"

    def shortname(self):
        return "atkinson"

    def val(self, **args):
        anchor_vectors = args['X'][args['anchor_indices']]
        distances_from_anchors = pairwise_distances(X=args['X'], Y=anchor_vectors, metric=dist, n_jobs=-1, **{'cost_funcs': args['cost_funcs'], 'cost_funcs_rev': args['cost_funcs_rev'], 
            'sens_group' : self.sens_group, 'feature_info': self.feature_info, 'users_mat': args['X'], 'users_gt': args['y']})
        area_membership = np.argmin(distances_from_anchors, axis=1)
        assert len(area_membership) == len(args['X'])

        self.N = len(np.unique(area_membership))
        atkinson_idx = 0
        for i in area_membership:
            mask = area_membership == i
            p_i = np.count_nonzero(self.sens_group[mask])/np.count_nonzero(mask)
            atkinson_idx += np.power(p_i, self.beta) * np.power(1 - p_i, 1 - self.beta)
        atkinson_idx /= (self.N * self.T * self.P)
        atkinson_idx = np.power(atkinson_idx, 1/(1 - self.beta))
        atkinson_idx = (self.P/(1 - self.P)) * atkinson_idx
        atkinson_idx = 1 - atkinson_idx
        print ("Atkinson Index: {}".format(atkinson_idx))
        return atkinson_idx


class Centralization(Segregation):

    def __init__(self, sens_group, feature_info):
        super(Centralization, self).__init__(sens_group, feature_info)

    def __str__(self):
        return "CI"

    def shortname(self):
        return "centralization"

    def val(self, **args):
        y_pred = args['y_pred']
        # cutoff = np.mean(y_pred)
        cutoff = np.mean(args['y'])
        mask = y_pred >= cutoff
        ci = np.count_nonzero(np.logical_and(mask, self.sens_group))/np.count_nonzero(self.sens_group)
        print ("Centralization Index: {}".format(ci))
        return ci

class Clustering(Segregation):

    def __init__(self, sens_group, feature_info):
        super(Clustering, self).__init__(sens_group, feature_info)

    def __str__(self):
        return "ACI"

    def shortname(self):
        return "ACI"

    def find_c_ij(self, X_i, X_j, args):
        d_ij = dist(**{'x_i': X_i, 'x_j': X_j, 'cost_funcs': args['cost_funcs'], 'cost_funcs_rev': args['cost_funcs_rev'], 
            'sens_group' : self.sens_group, 'feature_info': self.feature_info, 'users_mat': args['X'], 'users_gt': args['y']})
        return np.exp(-d_ij)

    def val(self, **args):
        labels = np.array(list(zip(*enumerate(args['y'])))[0])
        num_classes = len(np.unique(labels))
        num_sens, num_nosens = np.count_nonzero(self.sens_group), np.count_nonzero(~self.sens_group)
        a_00, a_01, a_10, a_11 = 0, 0, 0, 0

        C_ij = pairwise_distances(X=args['X'], metric=dist, n_jobs=-1, **{'cost_funcs': args['cost_funcs'], 'cost_funcs_rev': args['cost_funcs_rev'], 
            'sens_group' : self.sens_group, 'feature_info': self.feature_info, 'users_mat': args['X'], 'users_gt': args['y']})

        for i in range(num_classes):
            num_sens_i = np.count_nonzero(np.logical_and(labels == i, self.sens_group))
            b_00, b_10 = 0, 0
            for j in range(num_classes):
                num_sens_j = self.sens_group[j].astype(int)
                size_of_class_j = 1
                c_ij = C_ij[i,j]
                a_01 += c_ij
                a_11 += c_ij
                b_00 += c_ij * num_sens_j
                b_10 += c_ij * size_of_class_j
            a_00 += (num_sens_i/num_sens) * b_00
            a_10 += (num_sens_i/num_sens) * b_10
        a_01 += (num_sens/(num_classes**2)) * a_01
        a_11 += (num_sens/(num_classes**2)) * a_11

        aci = (a_00 - a_01) / (a_10 - a_11)
        print ("ACI: {}".format(aci))
        return aci