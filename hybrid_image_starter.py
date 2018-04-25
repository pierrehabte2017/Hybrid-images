import matplotlib.pyplot as plt
import numpy as np
from align_image_code import align_images
from align_image_code import hybrid_image
import scipy.misc

#First load images

# high sf
# im1 = plt.imread('julie.jpg')#/255.
# im1 = im1[:,:,:3]
#
# # low sf
# im2 = plt.imread('meryem.jpg')#/255
# im2 = im2[:,:,:3]
#
# # Next align images (this code is provided, but may be improved)
# im1_aligned, im2_aligned = align_images(im1, im2)
# scipy.misc.imsave('im1_aligned.png', im1_aligned)
# scipy.misc.imsave('im2_aligned.png', im2_aligned)



## You will provide the code below. Sigma1 and sigma2 are arbitrary
## cutoff values for the high and low frequencies

im1_aligned=plt.imread('im2_aligned.png')
im2_aligned=plt.imread('im1_aligned.png')
#param
sigma1 = 6
sigma2 = 0.1
Lambda= 0.6


hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2, Lambda)

plt.imshow(hybrid)
plt.show()

#we filter out the outsanding values
p=1/20
hybrid2=np.maximum(hybrid,p )

plt.imshow(hybrid2)
plt.show()




