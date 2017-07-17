import h5py
import os

import numpy as np
from scipy.misc import imresize


def calculate_mean():
    # resize the image so that its smaller size is 224
    # crop the middle 224x224 part 
    image_size = 224
    f_in = h5py.File('pascal_voc2007.h5', 'r')
    no_trainval = len(f_in['trainval_images'])

    f_out = h5py.File('temp.h5', 'w')
    image_h = f_out.create_dataset('trainval_images', (no_trainval, image_size, image_size, 3), dtype=np.float32)

    for ind in range(no_trainval):
        image = np.reshape(f_in['trainval_images'][ind], f_in['trainval_image_shapes'][ind])
                
        image = image.astype(np.float32)
        size_lower = min(image.shape[:2])
        image = imresize(image, np.float32(image_size) / size_lower)
        image = image[(image.shape[0] - image_size) / 2 :(image.shape[0] + image_size) / 2,
                    (image.shape[1] - image_size) / 2 :(image.shape[1] + image_size) / 2]
        image_h[ind] = image
    f_in.close()

    mean = np.array([np.mean(image_h[:,:,:,0]), np.mean(image_h[:,:,:,1]), np.mean(image_h[:,:,:,2])])

    f_out.close()
    os.remove('temp.h5')

    f_out = h5py.File('pascal_voc2007.h5', 'a')
    mean_h = f_out.create_dataset('mean', data=mean)
    f_out.close()