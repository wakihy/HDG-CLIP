import os
from .coco import CocoDetection
import torchvision.transforms as transforms
from randaugment import RandAugment


def create_dataset():
    path, dataset = '/data/yanjiexuan/coco/data/', 'coco'
    if 'coco' in dataset:
        instances_path_val = os.path.join(path, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(path, 'annotations/instances_train2014.json')
        
        data_path_val   = f'{path}/val2014'    
        data_path_train = f'{path}/train2014'  
        val_dataset = CocoDetection(data_path_val,
                                    instances_path_val,
                                    transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        # normalize, # no need, toTensor does normalization
                                    ]))

        train_dataset = CocoDetection(data_path_train,
                                          instances_path_train,
                                          transforms.Compose([
                                              transforms.Resize((224,224)),
                                              transforms.ToTensor(),
                                              # normalize,
                                          ]))

    
    return train_dataset, val_dataset




