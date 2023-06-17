'''
File created by Reza Kalantar - 29/11/2022
'''

import glob
import config
import random
from monai.data import DataLoader, CacheDataset, Dataset

def CreateDataloader(args, mode='train', shuffle=True, cache=False):
    '''
    params: 
    args: The arguments from argsparser after running main.py (including data_path, where the data is stored).
    mode: 'train' or 'test'
    shuffle: whether to shuffle data in the dataloader. Default=True.
    cache: wheter to use CacheDataset from Monai to speed up preprocessing and training. Default=False.
    
    return: return the torch-based dataloader for train images
    '''

    files_A = glob.glob(f'{args.data_path}/{mode}/A/*.nii.gz')
    random.shuffle(files_A) # shuffle files_A to ensure random patient selection for A and B
    files_B = glob.glob(f'{args.data_path}/{mode}/B/*.nii.gz')

    print(f"[INFO] {mode} A images: {len(files_A)}, {mode} B images: {len(files_B)}")

    #create dict for performing preprocessing transforms and creating dataloaders
    files_dict = [{'imgA': files_A[i], 'imgB': files_B[i]} for i in range(len(files_B))]

    if cache:
        ds = CacheDataset(data=files_dict, transform=config.train_transforms)
    else:
        ds = Dataset(data=files_dict, transform=config.train_transforms)
    data_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, pin_memory=True)
    return data_loader
