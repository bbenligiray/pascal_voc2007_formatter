# PASCAL VOC 2007 Formatter

A tool to download and format PASCAL VOC2007 dataset. It outputs a .h5 file that contains the following:

* data_types: 'train', 'val' and 'test'
* cats: names of the 20 categories

(replace x with any data type)

* x_images: flattened images (h5py allows differently sized elements only if they are 1D)
* x_shapes: shapes of the images, to reshape the flattened images
* x_names: file names of the images
* x_label: a one-hot integer vector of labels
