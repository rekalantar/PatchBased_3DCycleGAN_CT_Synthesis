'''
File created by Reza Kalantar - 29/11/2022
'''

import os
import time
import torch
import config
import argparse
from utils import *
from cyclegan3d import CycleGAN
from dataloader import CreateDataloader

parser = argparse.ArgumentParser()

parser.add_argument('data_path', type=str, help='location where the data is stored')
parser.add_argument('out_path', type=str, help='location where to save results')
parser.add_argument('max_iterations', default=1000, type=int, nargs='?', help='select the total number of iterations for training')
parser.add_argument('resume_training', default=False, type=bool, nargs='?', help='if resuming cycleGAN training. It also requires pretrained weights path')
parser.add_argument('pretrained_path', default='', type=str, nargs='?', help='pretrained weights path for loading the generator and discriminator')
parser.add_argument('save_train_freq', default=100, type=int, nargs='?', help='frequency to save training images')
parser.add_argument('save_weights_freq', default=200, type=int, nargs='?', help='frequency to save generator and discriminator weights')
parser.add_argument('batch_size', default=1, nargs='?', const=1,  type=str, help='batch size for training and testing')
parser.add_argument('g_residual_blocks', default=9, type=str, nargs='?', help='the number of residual blocks in the generator bottleneck')
parser.add_argument('lr_G', default=0.0002, nargs='?', const=1, help='generator learning rate')
parser.add_argument('lr_D', default=0.0002, nargs='?', const=1, help='discriminator learning rate')

args = parser.parse_args()


def main():
    print("[INFO] CycleGAN training initiated ...")
    train_data_loader = CreateDataloader(args, mode='train', shuffle=True, cache=True)
    # test_data_loader = CreateDataloader(args, mode='test', shuffle=False, cache=True)
    train_data_num = len(train_data_loader)

    date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    out_path = os.path.join(args.out_path, date_time)

    if not os.path.exists(os.path.join(out_path, 'images')):
        os.makedirs(os.path.join(out_path, 'images'))
        os.makedirs(os.path.join(out_path, 'saved_weights'))

    print(f'[INFO] the results will be saved to {date_time} directory ...')

    gan = CycleGAN(args)

    D_A_losses, D_B_losses, G_A2B_losses, G_B2A_losses, cycle_A_losses, cycle_B_losses = ([] for i in range(6))

    iteration = 0
    while iteration < args.max_iterations:
        loop_index = 0
        for batch_data in train_data_loader:
            imgA, imgB = batch_data["imgA"].detach().cpu().numpy()[0,...,None], batch_data["imgB"][0,...,None].detach().cpu().numpy()
            # print('imgA', imgA.shape, ' imgB', imgB.shape)

            fake_A, fake_B, cycle_A, cycle_B, G_A2B_loss, G_B2A_loss, cycle_A_loss, cycle_B_loss, D_A_loss, D_B_loss = gan.train_step(imgA, imgB)
        
            pred_slice=10 #slice number for saving patch slices during training
            if iteration % args.save_train_freq == 0:
                save_tmp_images(iteration, imgA[:,...,pred_slice,:],    imgB[:,...,pred_slice,:], 
                                           fake_A[:,...,pred_slice,:],  fake_B[:,...,pred_slice,:],
                                           cycle_A[:,...,pred_slice,:], cycle_B[:,...,pred_slice,:], 
                                           out_path #folder name where the predictions during training will be saved
                                )
            print(f'Iteration [{iteration}/{args.max_iterations}]', f'Loop index [{loop_index}/{train_data_num}]')

            D_A_losses.append(D_A_loss.numpy())
            D_B_losses.append(D_B_loss.numpy())
            G_A2B_losses.append(G_A2B_loss.numpy())
            G_B2A_losses.append(G_B2A_loss.numpy())
            cycle_A_losses.append(cycle_A_loss.numpy())
            cycle_B_losses.append(cycle_B_loss.numpy())

            if iteration % args.save_weights_freq == 0:
                gan.G_A2B.save_weights(f'{out_path}/saved_weights/{iteration}_G_A2B.h5')
                gan.G_B2A.save_weights(f'{out_path}/saved_weights/{iteration}_G_B2A.h5')
                gan.D_A.save_weights(f'{out_path}/saved_weights/{iteration}_D_A.h5')
                gan.D_B.save_weights(f'{out_path}/saved_weights/{iteration}_D_B.h5')

            loop_index += 1
            iteration += 1

if __name__ == "__main__":
    main()
