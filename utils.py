'''
File created by Reza Kalantar - 29/11/2022
'''

import cv2
import config
import numpy as np
import matplotlib.pyplot as plt

def truncateAndSave(real, synthetic, reconstructed, path_name):
        image = np.hstack((real, synthetic, reconstructed))
        image = image[:, :, 0] # remove channel index
        plt.imshow(image, cmap='gray')
        plt.grid(False)
        plt.axis('off')
        plt.savefig(path_name)
        plt.close()
        # image_min = np.min(image)
        # image_max = np.max(image)
        # image_scaled = 255 * (image - image_min) / (image_max - image_min)
        # cv2.imwrite(path_name, image_scaled)
    
def save_tmp_images(iteration, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B, 
                    reconstructed_image_A, reconstructed_image_B, out_path):
    '''
    Function to save images during training from the train_dataloader
    '''
    try:
        real_images = np.vstack((real_image_A[0], real_image_B[0]))
        synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
        reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

        truncateAndSave(real_images, synthetic_images, reconstructed_images,
                             '{}/{}_{}.png'.format(
                                 out_path, 'images/tmp', iteration))
    except: # Ignore if file is open
        pass
