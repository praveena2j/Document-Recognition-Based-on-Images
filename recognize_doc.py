import cv2
import numpy as np
import pickle
import os, sys
import argparse, logging
import datetime

from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import *
from sklearn.svm import SVC 
#from skimage.feature import hog
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#======   Parameters    =========
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
gUsage = "python recognize_doc.py --image (-i) --feature (-f) --classifer (-c)"
gWorkDir = cwd
goutputmodels = gWorkDir + '/TrainingModels/'
gLogFileName = gWorkDir + "Doc-classifier.log"

#========     Code for Doc-classifier =============

def getConfigParams():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image")
	ap.add_argument("-f", "--feature")
	ap.add_argument("-c", "--classifer")

	args = vars(ap.parse_args())
	imagePath = args["image"]
	feature_select = args["feature"]
	classifer_select = args["classifer"]
	if not feature_select:
		feature_select = "all"
	if not classifer_select:
		classifer_select = "all"
	if not imagePath:
		print gUsage
		sys.exit(0)
	return imagePath, feature_select, classifer_select
	

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

def compute_bow_features(file_name, labs):
	test_image = cv2.imread(file_name, 0)
	kpts = fea_det.detect(test_image)
	kpts, des = des_ext.compute(test_image, kpts)

	voc = np.load(goutputmodels + 'voc.npy')
	k = 25
	# Calculate the histogram of features
	im_feature = np.zeros(k, "float32")
	words, distance = vq(des,voc)
	for w in words:
		im_feature[w] += 1
	# Scaling the words
	stdSlr = StandardScaler().fit([im_feature])
	im_feature = stdSlr.transform([im_feature])
	return im_feature

def compute_lda_features(file_name, labs):
	test_image = preprocessing(file_name)
	testimage = test_image.flatten()
	loaded_lda_model = pickle.load(open(goutputmodels + 'LDA.pkl', 'rb'))
	lda_feature = loaded_lda_model.transform(testimage)
	return lda_feature

def compute_hog_features(file_name, labs):
	test_image = cv2.imread(file_name, 0)
	hog_feature = hog.compute(test_image,winStride,padding,locations)
	arr_hog = hog_feature.ravel()
	return arr_hog

def compute_lbph_features(file_name, lab):
	image = cv2.imread(file_name, 0)
	test_recognizer.train([image], np.array(int(lab[0])))
	hists = test_recognizer.getMatVector("histograms")
	lbph_feature = hists[0]
	return lbph_feature

def prediction_module(feature, select_classifier, feature_type):
	if select_classifier == 'nonlinearsvm':
		loaded_model = pickle.load(open(goutputmodels + 'SVMmodel_nonlinear_' + feature_type + '.pkl', 'rb'))
		predicted_label = loaded_model.predict(feature)
		predicted_scores = loaded_model.decision_function(feature)
		predicted_score = predicted_scores[0][predicted_label.astype(int)]
	elif select_classifier == 'gaussiannaivebayes':
		loaded_model = pickle.load(open(outputmodels + 'bayesnaive_gaussian_'+ feature_type +'.pkl', 'rb'))
		predicted_label = loaded_model.predict(feature)
	elif select_classifier == 'randomforest':
		loaded_model = pickle.load(open(goutputmodels + 'randomforest_'+ feature_type +'.pkl', 'rb'))
		predicted_label = loaded_model.predict(feature)
	elif select_classifier == 'adaboost':
		loaded_model = pickle.load(open(goutputmodels + 'adaboost_'+ feature_type +'.pkl', 'rb'))
		predicted_label = loaded_model.predict(feature)
	else:
		print "Classifier not found"
	return predicted_label

def recognizedoc_module(test_image, select_feature, select_classifier):
	logging.info("Entered Recognition Mode")
	labs = '0'
	if select_feature == 'HOG':
		hog_feature = compute_hog_features(test_image, labs)
		prediction_label = prediction_module(hog_feature, select_classifier, select_feature)
	elif select_feature == 'LBPH':	
		lbph_features = compute_lbph_features(test_image, labs)
		prediction_label = prediction_module(lbph_features, select_classifier, select_feature)
	elif select_feature == 'LDA':	
		lda_features = compute_lda_features(test_image, labs)
		prediction_label = prediction_module(lda_features, select_classifier, select_feature)
	elif select_feature == 'BOW':	
		bow_features = compute_bow_features(test_image, labs)
		prediction_label = prediction_module(bow_features, select_classifier, select_feature)
	print prediction_label

if __name__ == '__main__':
	startTime = datetime.datetime.now()
	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename = gLogFileName, level = logging.DEBUG)

	imagePath, select_feature, select_classifier = getConfigParams()

	logging.info("Input Image: " + imagePath)
	logging.info("Feature Used: " + select_feature)
	logging.info("Classifer Used: " + select_classifier)

	#file_names, labs = list_files("Data")
	recognizedoc_module(imagePath, select_feature, select_classifier)