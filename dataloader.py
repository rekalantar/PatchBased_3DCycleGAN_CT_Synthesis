'''
File created by Reza Kalantar - 29/11/2022
'''

import glob
import config
from monai.data import DataLoader, CacheDataset, Dataset

def CreateDataloader(args, mode='train', shuffle=True, cache=False):
    '''
    params: 
    args: The arguments from argsparser after running main.py
    mode: train or test
    data_path: path where .nii.gz files are stored
    
    return: return the torch-based dataloader for train images
    '''

    files_A = glob.glob(f'{args.data_path}/{mode}/A/*.nii.gz')
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
