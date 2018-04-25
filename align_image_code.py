import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import scipy
import matplotlib.image as mpimg

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)


def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = int(np.abs(2 * r + 1 - R))
    cpad = int(np.abs(2 * c + 1 - C))
    return np.pad(
        im, [(0 if r > (R - 1) / 2 else rpad, 0 if r < (R - 1) / 2 else rpad),
             (0 if c > (C - 1) / 2 else cpad, 0 if c < (C - 1) / 2 else cpad),
             (0, 0)], 'constant')


def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    len2 = np.sqrt((p4[1] - p3[1]) ** 2 + (p4[0] - p3[0]) ** 2)
    dscale = len2 / len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale)
    else:
        im2 = sktr.rescale(im2, 1. / dscale)
    return im1, im2


def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi)
    return im1, dtheta


def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2 - h1) / 2.)): int(-np.ceil((h2 - h1) / 2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1 - h2) / 2.)): int(-np.ceil((h1 - h2) / 2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2 - w1) / 2.)): int(-np.ceil((w2 - w1) / 2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1 - w2) / 2.)):int( -np.ceil((w1 - w2) / 2.)), :]
    assert im1.shape == im2.shape
    return im1, im2


def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


def kernel(l,sigma):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    Kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))

    Kernel = Kernel / np.sum(Kernel)
    return Kernel

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



def hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2, Lambda):
    size=25

    H, W, _ = im1_aligned.shape

    # Filter 1
    kernel1 = kernel(size,sigma1)
    #filter 2
    kernel2 = kernel(size, sigma2)

    #filter3
    kernel3=kernel(size, 3 )

    #convolution
    H_final = H + size - 1
    W_final = W + size - 1

    im1_filtered=np.zeros((H_final, W_final,3))
    im2_filtered=np.zeros((H_final, W_final,3))

    for i in range(3):

        im1_filtered[:, :, i]=scipy.signal.convolve2d (im1_aligned[:,:,i], kernel1)

        im2_filtered[:,:,i ]= scipy.signal.convolve2d (im2_aligned[:,:,i], kernel3-kernel2 )



    #image1
    plt.figure(1)

    plt.subplot(331)
    plt.imshow(im1_aligned)
    plt.title(" Image1 original ")

    plt.subplot(333)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(rgb2gray(im1_aligned))))))
    plt.title(" Frequency domain image1 before filtering")

    plt.subplot(335)
    plt.imshow(kernel1)
    plt.title('Low-pass kernel')

    plt.subplot(337)
    plt.imshow(im1_filtered)
    plt.title(" Image1 filtered")


    plt.subplot(339)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(rgb2gray(im1_filtered))))))
    plt.title(" Frequency domain image1 after low-pass filter")
    plt.show()

    #image2
    plt.figure(2)

    plt.subplot(331)
    plt.imshow(im2_aligned)
    plt.title(" Image2 original ")

    plt.subplot(333)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(rgb2gray(im2_aligned))))))
    plt.title(" Frequency domain image1 before filtering")

    plt.subplot(335)
    plt.imshow(kernel2)
    plt.title('high-pass kernel')

    plt.subplot(337)
    plt.imshow(im2_filtered)
    plt.title(" Image2 filtered")

    plt.subplot(339)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(rgb2gray(im2_filtered))))))
    plt.title(" Frequency domain image2 after high-pass filter")
    plt.show()

    # linear combination of bot images
    hybrid_img=im1_filtered*Lambda + (1-Lambda)*im2_filtered

    plt.figure(3)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(rgb2gray(hybrid_img))))))
    plt.title(" Frequency domain hybrid image ")
    plt.show()


    return hybrid_img


