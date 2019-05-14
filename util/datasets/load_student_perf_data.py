from sklearn.externals import joblib

DATASET_FOLDER = 'student_performance'

def load_student_perf_data(ds_name):
	if ds_name == "StudentPerf":
		ds = joblib.load('../util/datasets/data/{}/processed_student_por.pkl'.format(DATASET_FOLDER))
	elif ds_name == "StudentPerfMut":
		ds = joblib.load('../util/datasets/data/{}/processed_student_por_mutable.pkl'.format(DATASET_FOLDER))
	elif ds_name == "StudentPerfMutPlusImmut":
		ds = joblib.load('../util/datasets/data/{}/processed_student_por_allfts.pkl'.format(DATASET_FOLDER))
	return ds