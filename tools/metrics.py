import numpy as np
from math import log10, sqrt
import cv2
from skimage.metrics import structural_similarity

class metrics:
    def __init__(self,gen_img,gt_img):
        """
        input : array of preprocess image
        """
        self.gen_img = gen_img.astype(np.float32)
        self.gt_img = gt_img.astype(np.float32)
        self.max_pixel = 255.0

    def psnr(self):
        """
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        """
        mse = self.mse()
        if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                    # Therefore PSNR have no importance. 
            return 100
        psnr = 20.0 * log10(self.max_pixel) - 10 * log10(mse)
        return psnr 

    def mse(self):
        # mse = np.mean((self.gt_img - self.gen_img) ** 2) 
        mse = np.mean(np.square(self.gen_img - self.gt_img))
        return mse
    
    def ssim(self):
        k1 = 0.01
        k2 = 0.03
        L = self.max_pixel
        C1 = (k1*L)**2
        C2 = (k2*L)**2
        
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(self.gen_img, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(self.gt_img, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(self.gen_img**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(self.gt_img**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(self.gen_img * self.gt_img, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
        # return structural_similarity(self.gt_img, self.gen_img, multichannel=True)

    def high_pass_x_y(self,image):
        if(image.shape[2] == 4):
            x_var = image[:, :, 1:, :] - image[:, :, :-1, :] # pixel diff equation
            y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
        elif(image.shape[2] == 3):
            x_var = image[:, 1:, :] - image[:, :-1, :]
            y_var = image[1:, :, :] - image[:-1, :, :]

        return x_var, y_var
    
    def total_variation_loss(self):
        x_deltas_gen, y_deltas_gen = self.high_pass_x_y(self.gen_img)
        tv_gen =  np.sum(np.abs(x_deltas_gen)) + np.sum(np.abs(y_deltas_gen))
        x_deltas_gt, y_deltas_gt = self.high_pass_x_y(self.gt_img)
        tv_gt =  np.sum(np.abs(x_deltas_gt)) + np.sum(np.abs(y_deltas_gt))
        return [tv_gen, tv_gt]