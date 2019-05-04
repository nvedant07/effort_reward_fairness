from sklearn.externals import joblib

DATASET_FOLDER = 'crimes_and_communities'

def load_crimes_and_communities():
	ds = joblib.load('../util/datasets/data/{}/processed_crimes_and_communities.pkl'.format(DATASET_FOLDER))
	return ds