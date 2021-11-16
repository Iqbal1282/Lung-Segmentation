import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
import random 
random.seed(0)
import pandas as pd 
import albumentations as A
import matplotlib.pyplot as plt 

class SegmentationDataset(Dataset):
	def __init__(self, label_path= 'dataset/train.csv' ,subfolder = './dataset', feature_folder = 'images', label_folder = 'masks' ,img_size = (256, 256), transforms = None):
		df = pd.read_csv(label_path)
		df['image'] =df['image'].apply(lambda x: x.replace('.png', ''))
		self._image_folder_names = df['image'].values
		self._mask_image_names = df['label'].values
		self.datalength = len(self._image_folder_names)
		self.feature_folder = feature_folder
		self.label_folder = label_folder
		self.subfolder = subfolder

		self.img_size = img_size
		self.transforms = transforms 




	def __len__(self):
		return self.datalength

	def __getitem__(self, index):
		image_path = self.subfolder + '/' + self.feature_folder + '/' + self._image_folder_names[index]+'/image.png'
		assert os.path.exists(image_path), image_path 
		mask_path = self.subfolder +'/' + self.label_folder +'/'+self._mask_image_names[index]
		assert os.path.exists(mask_path), mask_path

		img = cv2.imread(image_path, 0)
		img = np.array(cv2.resize(img, self.img_size) , dtype = np.float32) 

		mask = cv2.imread(mask_path, 0)
		mask = np.array(cv2.resize(mask, self.img_size), dtype = np.uint8) 


		if self.transforms is not None:

			transformed = self.transforms(image = img, mask = mask)
			img = transformed['image']
			mask = transformed['mask']
			 
		#img = A.Normalize(mean=(0.456, 0.456), std=(0.225, 0.225))(image = img)['image']
		#print(img.shape)
		#print(mask.shape)
		# plt.imshow(img)
		# plt.show()
		# plt.imshow(mask)
		# plt.show()
		img = self.transform(img)
		img = self.NormalizeData(img)
		mask = np.expand_dims(mask, 0)



		return (img, mask)


	def transform(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		image = np.transpose(image, (2, 0, 1))
		return image 

	def NormalizeData(self ,data):
		return (data - np.min(data)) / (np.max(data) - np.min(data))




