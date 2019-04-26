import cv2
import numpy as np
import pickle

from sklearn.svm import LinearSVC 
from skimage.feature import hog

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
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

features_1 = []
features_0 = []
labels = []

for i in range(10):	
	image_0 = cv2.imread("0_Test/" + str(i+1), 0)
	image_1 = cv2.imread("1_Test/" + str(i+1), 0)
	hist_0 = hog.compute(image_0,winStride,padding,locations)
	hist_1 = hog.compute(image_1,winStride,padding,locations)
	arr_0 = np.array(hist_0)
	arr_flat_0 = arr_0.ravel()
	features_0.append(arr_flat_0)
	labels.append('0')
	arr_1 = np.array(hist_1)
	arr_flat_1 = arr_1.ravel()
	features_1.append(arr_flat_1)
	labels.append('1')
print len(arr_flat_0)
class_0_mean = np.mean(features_0, axis=0)
#class_1_mean = numpy.mean(features)
print len(class_0_mean)
class_1_mean = np.mean(features_1, axis=0)

mean = []
mean.append(class_0_mean)
mean.append(class_1_mean)
np.save('mean.npy', mean)
print class_0_mean
a = np.load('mean.npy')
print a[0]
exit()
distance = []
for i in range(10):
	dist = np.linalg.norm(features[i] - class_0_mean)
	distance.append(dist)
print distance

#clf = LinearSVC()
#clf.fit(features, labels)

#SVmtrainedmodel = 'SVMtrainmodel_Docclassifer.pkl'
#pickle.dump(clf, open(SVmtrainedmodel,"wb"))

test_image = cv2.imread("6")
test_hist = hog.compute(test_image,winStride,padding,locations)
test_arr = np.array(test_hist)
arr_flat_test = test_arr.ravel()

print np.linalg.norm(arr_flat_test - class_0_mean)

exit()
predicted = clf.predict([features[8]])

#print features 
print labels

#======  results =======
print predicted