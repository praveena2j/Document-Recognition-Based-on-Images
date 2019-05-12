import cv2
import numpy as np
import pickle
import os, sys
import argparse, logging
import datetime

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import *
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#======   Parameters    =========
train_recognizer = cv2.createLBPHFaceRecognizer()
test_recognizer = cv2.createLBPHFaceRecognizer()
winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
					histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors
fea_det = cv2.FeatureDetector_create("BRISK")
des_ext = cv2.DescriptorExtractor_create("BRISK")

winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

cwd = os.getcwd()
gUsage = "python train.py --data (-d)--nofclasses (-n) --feature (-f) --classifier (-c)"
gWorkDir = cwd
gLogFileName = gWorkDir + "Doc-classifier.log"
goutputmodels = gWorkDir + '/TrainingModels/'
goutputcmatrices = gWorkDir + '/ConfusionMatrices/'
#========     Code for Doc-classifier =============

def getConfigParams():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--data")
	ap.add_argument("-n", "--nofclasses")
	ap.add_argument("-f", "--feature")
	ap.add_argument("-c", "--classifier")

	args = vars(ap.parse_args())
	dataPath = args["data"]
	nofclasses = args["nofclasses"]
	feature_select = args["feature"]
	classifer_select = args["classifier"]
	
	if not nofclasses:
		nofclasses = "all"
	if not feature_select:	
		feature_select = "all"
	if not classifer_select:
		classifer_select = "all"
	if not dataPath:
		print gUsage
		sys.exit()
	return dataPath, nofclasses, feature_select, classifer_select

def createDir(aDirpath):
	# if directory does not exist create it
	if not os.path.exists(aDirpath):
		os.makedirs(aDirpath)

def preprocessing(image):
	img = cv2.imread(image, 0)	
	resized_img = cv2.resize(img, (1700, 1350)) 
	return resized_img
	
def list_files(dir, nofclasses):
	r = []
	lab = []	
	for root, dirs, files in os.walk(dir):
		if not nofclasses == 'all':
			dirs = dirs[0:int(nofclasses)]
			for direct in dirs:
				rootd = os.path.join(root, direct)
				for name in os.listdir(rootd):
					r.append(os.path.join(rootd, name))
					lab.append(rootd[5])
		else:
			for name in files:
				r.append(os.path.join(root, name))
				lab.append(root[5])

	return r, lab

def compute_nonlinearsvm_model(features, labels):
	nonlinearsvm_clf = SVC(kernel = 'rbf', random_state = 0, gamma = 10, C = 100, decision_function_shape='ovr')
	nonlinearsvm_clf.fit(features, labels)
	nonlinearsvm_clf.fit(np.asarray(features), np.array(labels))
	return nonlinearsvm_clf

def compute_linearsvm_model(features, labels):
	linearsvm_clf = LinearSVC()
	linearsvm_clf.fit(np.asarray(features), np.array(labels))

	return linearsvm_clf

def compute_gaussian_naive_model(features, labels):
	gnb = GaussianNB()
	gnb_clf = gnb.fit(np.asarray(features), np.array(labels))

	return gnb_clf

def compute_randomforest_model(features, labels):
	rf_model = RandomForestClassifier(n_estimators=10)
	rf_clf = rf_model.fit(np.asarray(features), np.array(labels))

	return rf_clf

def compute_adaboost_model(features, labels):
	adaboost_model = AdaBoostClassifier( n_estimators=60, learning_rate=1)
	adaboost_clf = adaboost_model.fit(np.asarray(features), np.array(labels))

	return adaboost_clf

def compute_bow_features(file_names, labs):
	des_list = []
	labels = []
	for (file, lab_) in zip(file_names, labs):
		train_image = cv2.imread(file, 0)
		kpts = fea_det.detect(train_image)
		kpts, des = des_ext.compute(train_image, kpts)

		des_list.append((file, des)) 
		labels.append(int(lab_))

	descriptors = des_list[0][1]
	for image_path, descriptor in des_list[1:]:
		descriptors = np.vstack((descriptors, descriptor)) 

		# Perform k-means clustering
	k = 25
	voc, variance = kmeans(descriptors, k, 1) 
	np.save(goutputmodels + 'voc.npy', voc)
	# Calculate the histogram of features
	im_features = np.zeros((len(file_names), k), "float32")
	for i in xrange(len(file_names)):
		words, distance = vq(des_list[i][1],voc)
		for w in words:
			im_features[i][w] += 1

	# Scaling the words
	stdSlr = StandardScaler().fit(im_features)
	im_features = stdSlr.transform(im_features)
	return im_features, labels

def compute_lda_features(file_names, labs):
	training_images = []
	lda_labels = []
	for (file, lab) in zip(file_names, labs):
		train_image = preprocessing(file)
		training_images.append(train_image.flatten())
		lda_labels.append(int(lab))
	sklearn_lda = LDA(solver = 'svd')
	lda_model = sklearn_lda.fit(training_images, np.array(lda_labels))
	pickle.dump(lda_model, open(goutputmodels + 'LDA.pkl',"wb"))
	lda_features = lda_model.transform(training_images)
	return lda_features, lda_labels

def compute_hog_features(file_names, labs):
	hog_features = []
	hog_labels = []
	for (file, lab) in zip(file_names, labs):
		train_image = cv2.imread(file, 0)
		hog_feature = hog.compute(train_image,winStride,padding,locations)
		arr = np.array(hog_feature)
		arr_hog = arr.ravel()
		hog_features.append(arr_hog)
		hog_labels.append(lab)
	return hog_features, hog_labels

def compute_lbph_features(file_names, labs):
	training_images = []
	lbph_labels = []
	for (file, lab) in zip(file_names, labs):
		train_image = cv2.imread(file, 0)
		training_images.append(train_image)
		lbph_labels.append(int(lab))
	train_recognizer.train(training_images, np.array(lbph_labels))
	hists = train_recognizer.getMatVector("histograms")

	lbph_features = []
	for i in range(len(hists)):
		lbph_features.append(hists[i][0])
	return lbph_features, lbph_labels


def confusionmatrices(classifer_model, features, labels, feature_type, classifier_type):
	pred = classifer_model.predict(features)
	cnf_matrix = confusion_matrix(labels, pred)
	np.savetxt(goutputcmatrices + classifier_type + feature_type + '.csv', cnf_matrix, delimiter = ',')

def classifer_models(features, labels, feature_type):
	
	nonlinearsvm_clf = compute_nonlinearsvm_model(features, labels)
	classifer_type = 'nonlinearsvm'
	confusionmatrices(nonlinearsvm_clf, features, labels, feature_type, classifer_type)
	#np.save(goutputcmatrices + 'nonlinearsvm_' + feature_type + '.npy', svm_cnf_matrix)
	pickle.dump(nonlinearsvm_clf, open(goutputmodels + 'SVMmodel_nonlinear_' + feature_type + '.pkl',"wb"))

	gnb_clf = compute_gaussian_naive_model(features, labels)
	classifer_type = 'gaussiannaivebayes'
	confusionmatrices(gnb_clf, features, labels, feature_type, classifer_type)
	pickle.dump(gnb_clf, open(goutputmodels + 'bayesnaive_gaussian_'+ feature_type +'.pkl',"wb"))

	rf_clf = compute_randomforest_model(features, labels)
	classifer_type = 'randomforest'
	confusionmatrices(rf_clf, features, labels, feature_type, classifer_type)
	pickle.dump(rf_clf, open(goutputmodels + 'randomforest_'+ feature_type +'.pkl',"wb"))
	
	adaboost_clf = compute_adaboost_model(features, labels)
	classifer_type = 'adaboost'
	confusionmatrices(adaboost_clf, features, labels, feature_type, classifer_type)
	pickle.dump(adaboost_clf, open(goutputmodels + 'adaboost_'+ feature_type +'.pkl',"wb"))

def classifierselect_module(features, labels, feature_type, select_classifier):
	if select_classifier == 'nonlinearsvm':
		nonlinearsvm_clf = compute_nonlinearsvm_model(features, labels)
		confusionmatrices(nonlinearsvm_clf, features, labels, feature_type, select_classifier)
		pickle.dump(nonlinearsvm_clf, open(goutputmodels + 'SVMmodel_nonlinear_' + feature_type + '.pkl',"wb"))
	elif select_classifier == 'naivebayes':
		gnb_clf = compute_gaussian_naive_model(features, labels)
		confusionmatrices(gnb_clf, features, labels, feature_type, select_classifier)
		pickle.dump(gnb_clf, open(goutputmodels + 'bayesnaive_gaussian_'+ feature_type +'.pkl',"wb"))
	elif select_classifier == 'randomforest':
		rf_clf = compute_randomforest_model(features, labels)
		confusionmatrices(rf_clf, features, labels, feature_type, select_classifier)
		pickle.dump(rf_clf, open(goutputmodels + 'randomforest_'+ feature_type +'.pkl',"wb"))
	elif select_classifier == 'adaboost':
		adaboost_clf = compute_adaboost_model(features, labels)
		confusionmatrices(adaboost_clf, features, labels, feature_type, select_classifier)
		pickle.dump(adaboost_clf, open(goutputmodels + 'adaboost_'+ feature_type +'.pkl',"wb"))

def featureselect_module(file_names, labs, select_feature, select_classifier):
	if select_feature == 'HOG':
		hog_features, hog_labels = compute_hog_features(file_names, labs)
		if select_classifier == 'all':
			classifer_models(hog_features, hog_labels, select_feature)
		else:
			classifierselect_module(hog_features, hog_labels, select_feature, select_classifier)
	elif select_feature == 'LBPH':	
		lbph_features, lbph_labels = compute_lbph_features(file_names, labs)
		if select_classifier == 'all':
			classifer_models(lbph_features, lbph_labels, select_feature)
		else:
			classifierselect_module(lbph_features, lbph_labels, select_feature, select_classifier)
	elif select_feature == 'LDA':	
		lda_features, lda_labels = compute_lda_features(file_names, labs)
		if select_classifier == 'all':
			classifer_models(lda_features, lda_labels, select_feature)
		else:
			classifierselect_module(lda_features, lda_labels, select_feature, select_classifier)
	elif select_feature == 'BOW':	
		bow_features, bow_labels = compute_bow_features(test_image, labs)
		if select_classifier == 'all':
			classifer_models(bow_features, bow_labels, select_feature)
		else:
			classifierselect_module(bow_features, bow_labels, select_feature, select_classifier)

def training_module(file_names, labs, select_feature, select_classifier):	
	logging.info("Entered Training Mode")

	if select_feature == 'all' and select_classifier == 'all':
		hog_features, hog_labels = compute_hog_features(file_names, labs)
		classifer_models(hog_features, hog_labels,'HOG')

		lbph_features, lbph_labels = compute_lbph_features(file_names, labs)
		classifer_models(lbph_features, lbph_labels,'LBPH')

		lda_features, lda_labels = compute_lda_features(file_names, labs)
		classifer_models(lda_features, lda_labels,'LDA')

		bow_features, bow_labels = compute_bow_features(file_names, labs)
		classifer_models(bow_features, bow_labels,'BOW')
	else:
		featureselect_module(file_names, labs, select_feature, select_classifier)

if __name__ == '__main__':
	startTime = datetime.datetime.now()
	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename = gLogFileName, level = logging.DEBUG)

	dataPath, nofclasses, select_feature, select_classifier = getConfigParams()

	logging.info("Data Path: " + dataPath)
	logging.info("Feature Used: " + select_feature)
	logging.info("Classifer Used: " + select_classifier)

	createDir(goutputmodels)
	createDir(goutputcmatrices)

	file_names, labs = list_files(dataPath, nofclasses)
	training_module(file_names, labs, select_feature, select_classifier)		