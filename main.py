import os
import shutil
import h5py

import numpy as np
from PIL import Image
import xml.etree.ElementTree


def main():
	# clean up
	if os.path.isdir('VOCdevkit'):
		shutil.rmtree('VOCdevkit')

	tar_file_names = ['VOCtest_06-Nov-2007', 'VOCtrainval_06-Nov-2007', 'VOCdevkit_08-Jun-2007']
	for tar_file_name in tar_file_names:
		# download the tar file if it is not there
		if not os.path.isfile(tar_file_name + '.tar'):
			os.system('wget -t0 -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/' + tar_file_name + '.tar')
		#extract the downloaded file
		os.system('tar -xf ' + tar_file_name + '.tar')

	data_types = ['train', 'val', 'test']
	# read the file names
	image_names = []
	no_images = []
	for data_type in data_types:
		with open(os.path.join('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', data_type + '.txt')) as f:
			image_names.append(f.read().splitlines())
			no_images.append(len(image_names[-1]))

	# read the images and write to .h5 file
	image_path = os.path.join('VOCdevkit', 'VOC2007', 'JPEGImages')

	f = h5py.File('pascal_voc2007.h5', 'w')

	dt_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
	dt_str = h5py.special_dtype(vlen=str)

	data_types_h = f.create_dataset('data_types', (len(data_types),), dtype=dt_str)
	for ind_data_type, data_type in enumerate(data_types):
		data_types_h[ind_data_type] = data_type
		
	# we want to store the images in their original sizes
	# h5py only supports variable sized 1D arrays
	# so we flatten the the images and store them with their shapes
	# they can be reshaped afterwards
	for ind_data_type, data_type in enumerate(data_types):
		image_h = f.create_dataset(data_type + '_images', (no_images[ind_data_type],), dtype=dt_uint8)
		name_h = f.create_dataset(data_type + '_image_names', (no_images[ind_data_type],), dtype=dt_str)
		shape_h = f.create_dataset(data_type + '_image_shapes', (no_images[ind_data_type], 3), dtype=np.int)
		for image_ind, image_name in enumerate(image_names[ind_data_type]):
			image = Image.open(os.path.join(image_path, image_name + '.jpg'))
			np_image = np.array(image)
			image_h[image_ind] = np_image.flatten()
			name_h[image_ind] = image_name
			shape_h[image_ind] = np_image.shape

	# write the categories
	cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
		'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
		'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
	cats_h = f.create_dataset('cats', (len(cats),), dtype=dt_str)
	for ind_cat, cat in enumerate(cats):
		cats_h[ind_cat] = cat

	# write the image labels
	ann_path = os.path.join('VOCdevkit', 'VOC2007', 'Annotations')
	for ind_data_type, data_type in enumerate(data_types):
		label_h = f.create_dataset(data_type + '_labels', (no_images[ind_data_type], len(cats)), dtype=np.int)
		for image_ind, image_name in enumerate(image_names[ind_data_type]):
			ann = xml.etree.ElementTree.parse(os.path.join(ann_path, image_name + '.xml')).getroot()
			objs = ann.findall('object/name')
			image_labels = np.zeros(len(cats), dtype=np.int)
			for obj in objs:
				cat_id = cats.index(obj.text)
				image_labels[cat_id] = 1
			label_h[image_ind] = image_labels
	f.close()
	shutil.rmtree('VOCdevkit')

	# show random images to test
	f = h5py.File('pascal_voc2007.h5', 'r')
	cats_h = f['cats']
	data_types_h = f['data_types']
	while True:
		ind_data_type = np.random.randint(0, len(data_types_h))
		data_type = data_types_h[ind_data_type]

		image_h = f[data_type + '_images']
		name_h = f[data_type + '_image_names']
		shape_h = f[data_type + '_image_shapes']
		label_h = f[data_type + '_labels']

		ind_image = np.random.randint(0, len(image_h))

		np_image = np.reshape(image_h[ind_image], shape_h[ind_image])
		image = Image.fromarray(np_image, 'RGB')
		image.show()

		print('Image type: ' + data_type)
		print('Image name: ' + name_h[ind_image])
		for ind_cat, cat in enumerate(cats_h):
			if label_h[ind_image][ind_cat] == 1:
				print cat
		raw_input("...")


if __name__ == "__main__":
    main()