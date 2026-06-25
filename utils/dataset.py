import os
import pickle
import h5py
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from glob import glob

from utils.transforms import build_transform

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict   



def build_dataloader_open(args):
    #test_image_filenames = load_dict('/data2/test.pkl')
    test_image_filenames=None
    transform = build_transform(False, args)
    val_dataset = ValDataset_openimages(args, transform)

    #train_image_names = load_dict('/data2/train.pkl')
    train_image_names=None
    transform = build_transform(True, args)
    train_dataset = TrainDataset_openimages(args,  transform)
    return train_dataset, val_dataset


def build_dataloader(args):

    val_img_names = load_dict(os.path.join(args.data_path, 'test_img_names.pkl'))
    test_image_filenames = val_img_names['img_names']
    transform = build_transform(False, args)
    val_dataset = ValDataset(args, test_image_filenames, transform)

    img_names = load_dict(os.path.join(args.data_path, 'img_names.pkl'))
    image_filenames = img_names['img_names']
    transform = build_transform(True, args)
    train_image_names = np.array(image_filenames)
    train_dataset = TrainDataset(args, train_image_names, transform)

    return train_dataset, val_dataset

class TrainDataset(Dataset):

    def __init__(self, args, image_names, transforms):
        self.src = args.data_path
        train_loc = os.path.join(self.src, 'features' ,'nus_wide_train.h5')
        self.train_features = h5py.File(train_loc, 'r')
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, idx):
        file_name = self.image_names[idx]
        t = file_name.split("_")
        path = os.path.join(self.src, "Flickr", "_".join(t[5:-2]), t[-2]+"_"+t[-1])
        img = Image.open(path).convert('RGB')
        inputs = self.transforms(img)

        #features = np.int32(self.train_features.get(file_name+'-features'))
        label = np.int32(self.train_features.get(file_name+'-labels'))
        label = torch.from_numpy(label).long()

        return inputs, label,path

    def __len__(self):
        return len(self.image_names)

class ValDataset(Dataset):

    def __init__(self, args, image_names, transforms):
        self.src = args.data_path
        train_loc = os.path.join(self.src, 'features' ,'nus_wide_test.h5')
        self.train_features = h5py.File(train_loc, 'r')
        self.image_names = image_names
        self.transforms = transforms
        
    def __getitem__(self, idx):
        file_name = self.image_names[idx]

        t = file_name.split("_")
        path = os.path.join(self.src, "Flickr", "_".join(t[5:-2]), t[-2]+"_"+t[-1])
        img = Image.open(path).convert('RGB')
        inputs = self.transforms(img)

        labels_1006 =  np.int32(self.train_features.get(file_name+'-labels_1006'))
        labels_81 =  np.int32(self.train_features.get(file_name+'-labels_81'))

        return inputs, labels_1006, labels_81, path

    def __len__(self):
        return len(self.image_names)


def build_inf_dataloader(args):

    transform = build_transform(False, args)
    return  Filelist(args.filelist, args.img_root, transform)

class Filelist(Dataset):

    def __init__(self, filelist, root, transforms):
        
        self.items = []
        self.root = root
        self.transforms = transforms
        filelist = open(filelist).readlines()
        for file in filelist:
            self.items.append(file.strip())

    def __getitem__(self, idx):

        item = self.items[idx]
        label = " ".join(item.split(" ")[1:])
        file_name = os.path.join(self.root, item.split(" ")[0])
        img = Image.open(file_name).convert('RGB')
        inputs = self.transforms(img)

        return inputs, label, file_name

    def __len__(self):
        return len(self.items)


class TrainDataset_openimages(Dataset):

    def __init__(self, args, transforms):

        self.h5_files = glob(os.path.join('/data2/liubeiyan/openimage/train_features', '*.h5'))
        files_to_remove = []

        for i in self.h5_files:
            with h5py.File(i, 'r') as h5f:
                count = len(h5f.keys())
                if count != 640:
                    files_to_remove.append(i)

        for file in files_to_remove:
            self.h5_files.remove(file)

        print(len(files_to_remove)) 

        self.transforms = transforms

    def __getitem__(self, idx):
        h5_file_index = idx // 640
        #print(idx)
        #print(len(self.h5_files))
        h5_file_path = self.h5_files[h5_file_index]
        self.train_features = h5py.File(h5_file_path, 'r')
        with h5py.File(h5_file_path, 'r') as h5_file:
            keys = list(h5_file.keys())
            file_name = keys[idx % 640][:-11]
            path = os.path.join('/data2/liubeiyan/openimage/image_data/train', file_name)
            img = Image.open(path).convert('RGB')
            inputs = self.transforms(img)

        label = np.int32(self.train_features.get(file_name + '-seenlabels'))
        label = torch.from_numpy(label).long()

        return inputs, label

    def __len__(self):
        return len(self.h5_files)*640

class ValDataset_openimages(Dataset):

    def __init__(self, args,transforms):
        train_loc = '/data2/liubeiyan/openimage/test_features/openimage_test_2.h5'
        self.train_features = h5py.File(train_loc, 'r')
        self.keys = list(self.train_features.keys())
        self.transforms = transforms
        df_top_unseen = pd.read_csv('/data2/liubeiyan/openimage/top_400_unseen.csv', header=None)
        self.idx_top_unseen = df_top_unseen.values[:, 0]

    #def __getitem__(self, idx):
        #path = self.image_names[idx]
        #img = Image.open(path).convert('RGB')
        #file_name = path.split('/')[-1]
        #inputs = self.transforms(img)

        #seen = np.int32(self.train_features.get(file_name + '-seenlabels'))
        #seen = torch.tensor(seen)
        ## print(seen.shape)
        #unseen = np.int32(self.train_features.get(file_name + '-unseenlabels'))
        #unseen_label = unseen[self.idx_top_unseen]
        #unseen_label = torch.tensor(unseen_label)
        ## print(unseen_label.shape)
        #all_label = torch.cat((seen, unseen_label), 0)
        #return inputs, all_label, unseen_label, file_name

    def __getitem__(self, idx):
        file_name = self.keys[idx*3][:-9]
        path = os.path.join('/data2/liubeiyan/openimage/image_data/test', file_name)
        img = Image.open(path).convert('RGB')
        inputs = self.transforms(img)
        seen = np.int32(self.train_features.get(file_name + '-seenlabels'))
        seen = torch.tensor(seen)
        # print(seen.shape)
        unseen = np.int32(self.train_features.get(file_name + '-unseenlabels'))
        unseen_label = unseen[self.idx_top_unseen]
        unseen_label = torch.tensor(unseen_label)
        # print(unseen_label.shape)
        all_label = torch.cat((seen, unseen_label), 0)
        return inputs, all_label, unseen_label, file_name

    def __len__(self):
        return len(self.keys)//3