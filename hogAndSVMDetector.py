import os
from skimage import data, color, exposure
from skimage.feature import hog
from skimage.transform import resize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib
from sklearn import svm
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import math
import cv2
import time


# class HogandSVM():
# 	def __init__():



class HoGandSVM:
	def __init__(self, img_size = [300,150],train_img_size = [112,92],hog_orientations = 9, hog_pixels_per_cell = (8,8), hog_cells_per_block = (3, 3), hog_visualise = False):
		self.img_size = img_size
		self.hog_orientations = hog_orientations
		self.hog_pixels_per_cell = hog_pixels_per_cell
		self.hog_cells_per_block = hog_cells_per_block
		self.hog_visualise = hog_visualise
		self.train_img_size = train_img_size
		self.detect_TP = 0
		self.detect_TN = 0
		self.detect_FP = 0
		self.detect_FN = 0

	def extractHoGFeatures(self,train_imgs_path, save_train_features_path, save_train_label_path):
		self.train_imgs_path = train_imgs_path
		self.train_img_files = os.listdir(train_imgs_path)
		

		pos_features = []
		neg_features = []

		#img_size = [self.img_size[1], self.img_size[0]]
		for train_img_file in self.train_img_files:
			if "cropped" in train_img_file:
				if "pos-" in train_img_file:
					pos_train_img = data.imread(self.train_imgs_path+"/"+train_img_file ,as_grey=True)
					pos_train_img = resize(pos_train_img, self.train_img_size)
					feature = hog(pos_train_img,orientations=self.hog_orientations, pixels_per_cell=self.hog_pixels_per_cell,cells_per_block=self.hog_cells_per_block, visualise=self.hog_visualise)
					pos_features.append(feature)

				elif "neg-" in train_img_file:
					neg_train_img = data.imread(self.train_imgs_path+"/"+train_img_file, as_grey=True)
					neg_train_img = resize(neg_train_img, self.train_img_size)
					feature = hog(neg_train_img,orientations=self.hog_orientations, pixels_per_cell=self.hog_pixels_per_cell,cells_per_block=self.hog_cells_per_block, visualise=self.hog_visualise)
					neg_features.append(feature)

		pos_features = np.array(pos_features)

		neg_features = np.array(neg_features)

		train_features = np.append(pos_features,neg_features,axis=0)
		
		train_label = np.append(np.ones(pos_features.shape[0]),np.zeros(neg_features.shape[0]))
		np.savetxt(save_train_features_path, train_features, fmt="%f",delimiter=",")
		np.savetxt(save_train_label_path, train_label, fmt="%f",delimiter=",")
	
	def extractPosAndNegHoGFeatures(self,train_pos_imgs_path, train_neg_imgs_path, save_train_features_path, save_train_label_path):
		train_pos_img_files = os.listdir(train_pos_imgs_path)
		train_neg_img_files = os.listdir(train_neg_imgs_path)
		train_neg_imgs_path2 = "./att_faces"
		train_neg_img_files2 = os.listdir(train_neg_imgs_path2)
		

		pos_features = []
		neg_features = []
		#img_size = [self.img_size[1], self.img_size[0]]
		for train_img_file in train_pos_img_files:
			if ".pgm" in train_img_file and "croppedpos" in train_img_file:
				
				pos_train_img = data.imread(train_pos_imgs_path+"/"+train_img_file ,as_grey=True)
				pos_train_img = resize(pos_train_img, self.train_img_size)
				feature = hog(pos_train_img,orientations=self.hog_orientations, pixels_per_cell=self.hog_pixels_per_cell,cells_per_block=self.hog_cells_per_block, visualise=self.hog_visualise)
				pos_features.append(feature)
		for train_img_file in train_neg_img_files:
			if ".pgm" in train_img_file:
				
				neg_train_img = data.imread(train_neg_imgs_path+"/"+train_img_file ,as_grey=True)
				neg_train_img = resize(neg_train_img, self.train_img_size)
				feature = hog(neg_train_img,orientations=self.hog_orientations, pixels_per_cell=self.hog_pixels_per_cell,cells_per_block=self.hog_cells_per_block, visualise=self.hog_visualise)
				neg_features.append(feature)

		for train_img_file in train_neg_img_files2:
			if ".pgm" in train_img_file:
				
				neg_train_img = data.imread(train_neg_imgs_path2+"/"+train_img_file ,as_grey=True)
				neg_train_img = resize(neg_train_img, self.train_img_size)
				feature = hog(neg_train_img,orientations=self.hog_orientations, pixels_per_cell=self.hog_pixels_per_cell,cells_per_block=self.hog_cells_per_block, visualise=self.hog_visualise)
				neg_features.append(feature)

		pos_features = np.array(pos_features)
		neg_features = np.array(neg_features)

		train_features = np.append(pos_features,neg_features,axis=0)
		
		train_label = np.append(np.ones(pos_features.shape[0]),np.zeros(neg_features.shape[0]))
		np.savetxt(save_train_features_path, train_features, fmt="%f",delimiter=",")
		np.savetxt(save_train_label_path, train_label, fmt="%f",delimiter=",")

	def SVMFitting(self, features_path = "", label_path = "",dump_path = "Best_detector.pkl", grid_search_parm = [{'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}]):
		if features_path == "" or label_path == "":
			print("Please input features and label.")
		
		features = np.loadtxt(features_path, delimiter=",")
		label = np.loadtxt(label_path, delimiter=",")
		print ('start Grid Search')
		gscv = GridSearchCV(svm.LinearSVC(),grid_search_parm)
		gscv.fit(features, label)
		svm_best = gscv.best_estimator_
		print("Best Estimator: "+str(gscv.best_estimator_))
		print ('start re-learning SVM with best parameter set.')
		svm_best.fit(features, label)
		print ('finish learning SVM　with Grid-search.')
		joblib.dump(svm_best, dump_path, compress=9)

	#detect face by img_path
	def Detect(self, test_img_path,dump_path = "Best_detector.pkl", window_size = [112, 92],step_size = [5,5],show = True):
		print ('start loading SVM.')
		detector = joblib.load(dump_path)
		print ('finish loading SVM')
		win_w, win_h = window_size
		step_w,step_h = step_size
		
		test_img = data.imread(test_img_path,as_grey=True)
		test_img = np.array(test_img)
		img_size =[test_img.shape[0],test_img.shape[1]]
		img_w,img_h = [img_size[1], img_size[0]]
		print(test_img.shape)
		
		#win_size < img_size is needed 
		mgnf = min(float(img_w)/win_w, float(img_h)/win_h)
		if(mgnf < 1):
			img_size =[int(test_img.shape[0]*(0.01 + 1/mgnf)), int(test_img.shape[1]*(0.01 + 1/mgnf))]
			img_w,img_h = [img_size[1], img_size[0]]
			mgnf = min(float(img_w)/win_w, float(img_h)/win_h)
			test_img = resize(test_img, img_size)

		elif(mgnf>2):
			img_size =[int(test_img.shape[0]*(2/mgnf)), int(test_img.shape[1]*(2/mgnf))]
			img_w,img_h = [img_size[1], img_size[0]]
			mgnf = min(float(img_w)/win_w, float(img_h)/win_h)
			test_img = resize(test_img, img_size)
		k=0
		print(test_img.shape)
		#for scalability
		if(mgnf > 1.75):
			mgnfs = [0.5,0.6,0.75,1.0,1.25,1.5,mgnf]
		elif(mgnf > 1.5):
			mgnfs = [0.5,0.6,0.75,1.0,1.25,mgnf]
		else:
			mgnfs = [0.5,0.6,0.75,1.0,mgnf]
		window_size = np.array(window_size)
		while(k < len(mgnfs)):
			detected_list = []
			win_w, win_h = (window_size*mgnfs[k]).astype(int)
			print("window_size: " + str(win_w)+", " + str(win_h))
			for i in range(5):
				for x in range(0,img_w-step_w-win_w,step_w):
				    for y in range(0,img_h-step_h-win_h,step_h):
				    	window = test_img[y:y+win_h,x:x+win_w]
				    	window = resize(window, self.train_img_size)
				    	hogfeature = hog(window, orientations = self.hog_orientations, pixels_per_cell = self.hog_pixels_per_cell, cells_per_block = self.hog_cells_per_block, visualise = self.hog_visualise )
				    	hogfeature=np.array([hogfeature])
				    	estimated_class = 1/(1+(math.exp(-1*detector.decision_function(hogfeature))))
				    	if estimated_class >= 0.48: 
				    		detected_list.append([x,y,x+win_w,y+win_h])
			detected_list = self.NMS(detected_list)
			if len(detected_list) > 0:
				ss = 0
				for rect in detected_list:
					cv2.rectangle(test_img, tuple(rect[0:2]), tuple(rect[2:4]), (0,0,0), 2)
					write_path = "." + test_img_path.split(".")[1]+str(ss) +"ans.pgm"
					ss+=1
					Image.fromarray(np.uint8(test_img[rect[1]:rect[3],rect[0]:rect[2]]*255)).save(write_path)
				break
			k+=1
		if not (len(detected_list) > 0):
			write_path = "." + test_img_path.split(".")[1] +"ans.pgm"
			Image.fromarray(np.uint8(test_img*255)).save(write_path)


		if show:

			plt.subplot(111).set_axis_off()
			plt.imshow(test_img, cmap=plt.cm.gray)
			plt.title('Result')
			plt.show()

			
			#cv2.imwrite(write_path, test_img)

			return 0

		else:
			return detected_list

	#detect face by img
	def Detectimg(self, img,dump_path = "Best_detector.pkl", window_size = [112, 92],step_size = [5,5],show = True):
		print ('start loading SVM.')
		detector = joblib.load(dump_path)
		print ('finish loading SVM')
		win_w, win_h = window_size
		img_w,img_h = self.img_size
		step_w,step_h = step_size

		test_img = img
		img_size = [self.img_size[1], self.img_size[0]]
		test_img = resize(test_img, img_size)
		detected_list = []


		for i in range(5):
			for x in range(0,img_w-step_w-win_w,step_w):
			    for y in range(0,img_h-step_h-win_h,step_h):
			        window = test_img[y:y+win_h,x:x+win_w]
			        window = resize(window, self.train_img_size)
			        hogfeature = hog(window, orientations = self.hog_orientations, pixels_per_cell = self.hog_pixels_per_cell, cells_per_block = self.hog_cells_per_block, visualise = self.hog_visualise )
			        hogfeature=np.array([hogfeature])
			        estimated_class = 1/(1+(math.exp(-1*detector.decision_function(hogfeature))))
			        if estimated_class >= 0.48: 
			            detected_list.append([x,y,x+win_w,y+win_h])
		detected_list = self.NMS(detected_list)
		if len(detected_list) > 0:
			for rect in detected_list:
				cv2.rectangle(test_img, tuple(rect[0:2]), tuple(rect[2:4]), (0,0,0), 2)
		
		if show:

			plt.subplot(111).set_axis_off()
			plt.imshow(test_img, cmap=plt.cm.gray)
			plt.title('Result')
			plt.show()
			return 0

		else:
			return detected_list
	#save captured data, face size(if there is face) and normal size
	def capture(self,capture_path,img,itr,pos_or_neg = "pos",train_or_test = "train",only_cap = False):
		if not os.path.exists(capture_path):
			os.mkdir(capture_path)

		capture_path = capture_path + "/" + train_or_test
		if not os.path.exists(capture_path):
			os.mkdir(capture_path)

		#if you want to crop your face later, you should define only_cap = True.
		if not only_cap:
			detected_list =self.Detectimg(img,dump_path = "./HumansData/Best_humans_detector.pkl", window_size = [112, 92],step_size = [5,5],show = False)
			test_img = img
			img_size = [self.img_size[1], self.img_size[0]]
			test_img = resize(test_img, img_size)
			if (len(detected_list) != 0 and pos_or_neg == "pos"):
				for rect in detected_list:
					# plt.imshow(test_img[rect[1]:rect[3],rect[0]:rect[2]], cmap=plt.cm.gray)
					# plt.title('Result')
					# plt.show()
					# print(test_img)
					Image.fromarray(np.uint8(test_img[rect[1]:rect[3],rect[0]:rect[2]]*255)).save(capture_path +"/" + "cropped" + pos_or_neg + "-" +str(itr) + ".pgm")
			elif(pos_or_neg == "neg"):
				Image.fromarray(np.uint8(test_img[25:117,100:212]*255)).save(capture_path +"/" + "cropped" + pos_or_neg + "-" +str(itr) + ".pgm")
		write_path = capture_path +"/" + pos_or_neg + "-" +str(itr) + ".pgm"
		cv2.imwrite(write_path, img)

	#Non-Maximum-suppression
	def NMS(self,detected_list,overlap_th=0.6):
		detected_list = np.array(detected_list)
		if len(detected_list) == 0:
			return []

		if detected_list.dtype.kind == "i":
			detected_list = detected_list.astype("float")
 
		# initialize the list of picked indexes	
		selected_inds = []
	 
		x1 = detected_list[:,0]
		y1 = detected_list[:,1]
		x2 = detected_list[:,2]
		y2 = detected_list[:,3]
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		inds = np.argsort(y2)

		while len(inds) > 0:
			last = len(inds) - 1
			i = inds[last]
			selected_inds.append(i)
			xx1 = np.maximum(x1[i], x1[inds[:last]])
			yy1 = np.maximum(y1[i], y1[inds[:last]])
			xx2 = np.minimum(x2[i], x2[inds[:last]])
			yy2 = np.minimum(y2[i], y2[inds[:last]])

			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)
	 
			# compute the ratio of overlap
			overlap = (w * h) / area[inds[:last]]
	
			inds = np.delete(inds, np.concatenate(([last],
				np.where(overlap > overlap_th)[0])))

		return detected_list[selected_inds].astype("int")

	#capturing your face
	def videoCapture(self,capture_path,num_of_max_sample = 100, pos_or_neg = "pos",train_or_test = "train",only_cap = False):
		cap = cv2.VideoCapture(0)
		i=0
		while(i < num_of_max_sample):
		    ret, frame = cap.read()

		    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		    threshold = 70
		    max_pixel = 255
		    if(i<num_of_max_sample):
		    	self.capture(capture_path,gray,i,pos_or_neg,train_or_test,only_cap)


		    cv2.imshow('frame',gray)
		    
		    i += 1
		    if cv2.waitKey(500) & 0xFF == ord('q'):
		        break
		cap.release()
		cv2.destroyAllWindows()

	#to detect faces in multiple pictures
	def evaluate(self,test_pos_imgs_path, test_neg_imgs_path, dump_path = "./HumanData2/Best_human_detector.pkl"):
		test_pos_img_files = os.listdir(test_pos_imgs_path)
		
		for test_img_file in test_pos_img_files:
			if ".pgm" in test_img_file:
				test_img_path = test_pos_imgs_path+"/"+test_img_file
				self.Detect(test_img_path,dump_path = dump_path,window_size = [112,92])
				
		test_neg_img_files = os.listdir(test_neg_imgs_path)
		for test_img_file in test_neg_img_files:
			if ".pgm" in test_img_file:
				test_img_path = test_neg_imgs_path+"/"+test_img_file
				self.Detect(test_img_path,dump_path = dump_path,window_size = [112,92])






		


#to detect my face
def main():
	capture_path = "HumanData2"
	train_imgs_path = "./HumanData2/train"
	test_imgs_path = "./HumanData2/test/pos"
	save_train_features_path = "./HumanData2/human_train_features.csv"
	save_train_label_path = "./HumanData2/human_train_label.csv"
	test_img_path = './HumanData2/test/pos/pos-1.pgm'
	dump_path = "./HumanData2/Best_human_detector2.pkl"
	train_pos_imgs_path = "./HumanData2/train"
	train_neg_imgs_path = "./CarData/TrainImages"

	test_pos_imgs_path = test_imgs_path 
	test_neg_imgs_path = "./HumanData2/test/neg"
	num_of_max_sample = 20
	Detector = HoGandSVM()
	print("Capturing positive images...")
	Detector.videoCapture(capture_path,num_of_max_sample, pos_or_neg = "pos",train_or_test = "train")
	print("finish Capturing positive images...")
	time.sleep(3)

	print("Capturing negative images...")
	Detector.videoCapture(capture_path,num_of_max_sample, pos_or_neg = "neg",train_or_test = "train")
	print("finish Capturing negative images...")
	time.sleep(3)

	print("Capturing test images...")
	Detector.videoCapture(capture_path,num_of_max_sample, pos_or_neg = "pos",train_or_test = "test",only_cap = True)
	print("finish Capturing test images...")
	Detector.extractPosAndNegHoGFeatures(train_pos_imgs_path, train_neg_imgs_path, save_train_features_path, save_train_label_path)
	Detector.SVMFitting(features_path=save_train_features_path,label_path=save_train_label_path, dump_path = dump_path)
	#Detector.Detect(test_img_path,dump_path = dump_path,window_size = [112,92])
	Detector.evaluate(test_pos_imgs_path, test_neg_imgs_path)

# if you want to crop your face later
def main2():
	capture_path = "HumanData2"
	# print("Capturing positive images...")
	# Detector.videoCapture(capture_path,num_of_max_sample, pos_or_neg = "pos",train_or_test = "train",only_cap = True)
	# print("finish Capturing positive images...")
	# time.sleep(3)

	# print("Capturing negative images...")
	# Detector.videoCapture(capture_path,num_of_max_sample, pos_or_neg = "neg",train_or_test = "train",only_cap = True)
	# print("finish Capturing negative images...")
	# time.sleep(3)

	# print("Capturing test images...")
	# Detector.videoCapture(capture_path,2, pos_or_neg = "pos",train_or_test = "test",only_cap = True)
	# print("finish Capturing test images...")
	train_imgs_path = "./HumanData2/train"
	train_img_files = os.listdir(train_imgs_path)
	for train_img_file in train_pos_img_files:
		if "pos-" in train_img_file:
			img = data.imread(train_imgs_path+"/"+train_img_file ,as_grey=True)
		capture_path = capture_path + "/" + train_or_test
		if not os.path.exists(capture_path):
			os.mkdir(capture_path)
		detected_list =self.Detectimg(img,dump_path = "./HumansData/Best_humans_detector.pkl", window_size = [112, 92],step_size = [5,5],show = False)
	test_img = img
	img_size = [self.img_size[1], self.img_size[0]]
	test_img = resize(test_img, img_size)
	if (len(detected_list) != 0 and pos_or_neg == "pos"):
		for rect in detected_list:
			# plt.imshow(test_img[rect[1]:rect[3],rect[0]:rect[2]], cmap=plt.cm.gray)
			# plt.title('Result')
			# plt.show()
			# print(test_img)
			Image.fromarray(np.uint8(test_img[rect[1]:rect[3],rect[0]:rect[2]]*255)).save(capture_path +"/" + "cropped" + pos_or_neg + "-" +str(itr) + ".pgm")
	elif(pos_or_neg == "neg"):
		Image.fromarray(np.uint8(test_img[25:117,100:212]*255)).save(capture_path +"/" + "cropped" + pos_or_neg + "-" +str(itr) + ".pgm")

def HumanFaceDetect():
	#capture_path = "HumanData"
	save_train_features_path = "./HumansData/human_train_features.csv"
	save_train_label_path = "./HumansData/human_train_label.csv"
	test_img_path = './HumanData/test/pos-1.pgm'
	dump_path = "./HumansData/Best_humans_detector.pkl"
	num_of_max_sample = 100

	train_pos_imgs_path = "./att_faces"
	train_neg_imgs_path = "./CarData/TrainImages"

	Detector = HoGandSVM()
	Detector.extractPosAndNegHoGFeatures(train_pos_imgs_path, train_neg_imgs_path, save_train_features_path, save_train_label_path)
	Detector.SVMFitting(features_path=save_train_features_path,label_path=save_train_label_path, dump_path = dump_path)
	Detector.Detect(test_img_path,dump_path = dump_path,window_size = [112,92])
	

if __name__ == "__main__":
	main()
	











