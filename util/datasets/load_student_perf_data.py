from sklearn.externals import joblib

DATASET_FOLDER = 'student_performance'

def load_student_perf_data():
	ds = joblib.load('../util/datasets/data/{}/processed_student_por.pkl'.format(DATASET_FOLDER))
	return ds