#plot all the points and save as image
import numpy as np
import matplotlib.pyplot as plt
import mrcfile


conv_image = np.load('/nrs/liza/gold-aligner/conv_image.npy')
mrcfile.new("/nrs/liza/gold-aligner/gold_conv_images/gold_convolution_layer.mrc", data=conv_image.astype('float32'), overwrite = True) 
for z, layer in enumerate(conv_image):
    plt.imshow(layer, cmap = 'gray_r')
    plt.savefig(f"/nrs/liza/gold-aligner/gold_conv_images/gold_convolution_layer_{z}")


