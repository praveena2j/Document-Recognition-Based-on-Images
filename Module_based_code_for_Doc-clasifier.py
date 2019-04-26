import cv2
import numpy as np
import pickle
import os, sys
import argparse, logging
import datetime

from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import *
from sklearn.svm import LinearSVC 
#from skimage.feature import hog
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
gUsage = "python Doc-classifer.py --train (-t)--image (-i) --feature (-f) --classifer (-c)"
gWorkDir = cwd
gLogFileName = gWorkDir + "Doc-classifier.log"

#========     Code for Doc-classifier =============

def getConfigParams():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--train")
	ap.add_argument("-i", "--image")
	ap.add_argument("-f", "--feature")
	ap.add_argument("-c", "--classifer")

	args = vars(ap.parse_args())
	training_flag = args["train"]
	imagePath = args["image"]
	feature_select = args["feature"]
	classifer_select = args["classifer"]

	if training_flag == '1':
		imagePath = str(0)
		training_flag = str(1)
		feature_select = str(0)
		classifer_select = str(0)
		return training_flag, imagePath, feature_select, classifer_select
	elif training_flag == '0':
		training_flag = str(0)
		return training_flag, imagePath, feature_select, classifer_select
	else:
		print gUsage
        sys.exit(0);

def preprocessing(image):
	img = cv2.imread(image, 0)	
	resized_img = cv2.resize(img, (1700, 1350)) 
	return resized_img
	
def list_files(dir):
	r = []
	lab = []
	for root, dirs, files in os.walk(dir):
		for name in files:
			r.append(os.path.join(root, name))
			lab.append(root[5])
	return r, lab

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

	if (len(des_list) == 1):
		descriptors = des_list[0][1]
		voc = np.load('voc.npy')
	else:			
		descriptors = des_list[0][1]
		for image_path, descriptor in des_list[1:]:
			descriptors = np.vstack((descriptors, descriptor)) 

		# Perform k-means clustering
		k = 25
		voc, variance = kmeans(descriptors, k, 1) 

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
	lda_features = sklearn_lda.fit(training_images, np.array(lda_labels)).transform(training_images)
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

def classifer_models(features, labels, feature_type):
	linearsvm_clf = compute_linearsvm_model(features, labels)
	pickle.dump(linearsvm_clf, open('SVMmodel_linear_' + feature_type + '.pkl',"wb"))
	gnb_clf = compute_gaussian_naive_model(features, labels)
	pickle.dump(gnb_clf, open('bayesnaive_gaussian_'+ feature_type +'.pkl',"wb"))
	rf_clf = compute_randomforest_model(features, labels)
	pickle.dump(rf_clf, open('randomforest_'+ feature_type +'.pkl',"wb"))
	adaboost_clf = compute_adaboost_model(features, labels)
	pickle.dump(adaboost_clf, open('adaboost_'+ feature_type +'.pkl',"wb"))

def training_module(file_names, labs):	
	logging.info("Entered Training Mode")
	hog_features, hog_labels = compute_hog_features(file_names, labs)
	classifer_models(hog_features, hog_labels,'hog')

	lbph_features, lbph_labels = compute_lbph_features(file_names, labs)
	classifer_models(lbph_features, lbph_labels,'lbph')

	lda_features, lda_labels = compute_lda_features(file_names, labs)
	classifer_models(lda_features, lda_labels,'lda')

	bow_features, bow_labels = compute_bow_features(file_names, labs)
	classifer_models(bow_features, bow_labels,'bow')

def prediction_module(feature, select_classifier):
	if select_classifier == 'linearsvm':
		loaded_model = pickle.load(open('SVMmodel_linear_hog.pkl', 'rb'))
		predicted_label = loaded_model.predict(hog_features)
		predicted_scores = loaded_model.decision_function(hog_features)
		predicted_score = predicted_scores[0][predicted_label.astype(int)]
	elif select_classifier == 'naivebayes':
		loaded_model = pickle.load(open('bayesnaive_gaussian_hog.pkl', 'rb'))
		predicted_label = loaded_model.predict(hog_features)
	elif select_classifier == 'randomforest':
		loaded_model = pickle.load(open('randomforest_hog.pkl', 'rb'))
		predicted_label = loaded_model.predict(hog_features)
	elif select_classifier == 'adaboost':
		loaded_model = pickle.load(open('adaboost_hog.pkl', 'rb'))
		predicted_label = loaded_model.predict(hog_features)
	else:
		print "Classifier not found"
	return predicted_label

def recognize_module(test_image, select_feature, select_classifier):
	logging.info("Entered Recognition Mode")
	labs = '0'
	if select_feature == 'HOG':
		hog_features, hog_labels = compute_hog_features(test_image, labs)
		prediction_label = prediction_module(hog_features, select_classifier)
	elif select_feature == 'LBPH':	
		lbph_features, lbph_labels = compute_lbph_features(test_image, labs)
		prediction_label = prediction_module(lbph_features, select_classifier)
	elif select_feature == 'LDA':	
		lda_features, lda_labels = compute_lda_features(test_image, labs)
		prediction_label = prediction_module(lda_features, select_classifier)
	elif select_feature == 'BOW':	
		bow_features, bow_labels = compute_bow_features(test_image, labs)
		prediction_label = prediction_module(bow_features, select_classifier)
	print predicted_label

if __name__ == '__main__':
	startTime = datetime.datetime.now()
	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename = gLogFileName, level = logging.DEBUG)

	training_flag, imagePath, select_feature, select_classifier = getConfigParams()

	logging.info("Input Image: " + imagePath)
	logging.info("Feature Used: " + select_feature)
	logging.info("Classifer Used: " + select_classifier)

	if training_flag == "1":
		file_names, labs = list_files("Data")
		training_module(file_names, labs)		
	else:
		recognize_module(imagePath, select_feature, select_classifier)
