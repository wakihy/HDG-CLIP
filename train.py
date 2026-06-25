import os
import argparse
import datetime


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import clip
from utils.misc import *
from utils.src.dataset.create_dataset import *
from utils.dataset import build_dataloader
from utils.optimizer import build_optimizer
from models.clip_vit import CLIPVIT
from engine import test, train
from clip import *
import shutil

from utils.losses import create_loss
def main(args):

    setup_seed(args.seed)
    args.device = "cuda:"+str(args.divices) if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = True

    # Init Recoder
    record_name = datetime.datetime.now().strftime('%m-%d-%H:%M:%S') + "_" + "HDG-CLIP"
    args.record_path = os.path.join('logger', record_name)
    os.makedirs(args.record_path, exist_ok=True)
    logger = init_log(args, args.record_path)
    write_description_to_folder(os.path.join(args.record_path, "configs.txt"), args)

    current_file_path = os.path.abspath(__file__)
    directory_path = os.path.dirname(current_file_path)
    #print(directory_path)
    shutil.copyfile(directory_path+'/engine.py', args.record_path+'/engine.py')
    shutil.copyfile(directory_path+'/train.py', args.record_path+'/train.py')
    shutil.copyfile(directory_path+'/models/clip_vit.py', args.record_path+'/clip_vit.py')

    # Init DataLoader
    train_dataset, val_dataset = build_dataloader(args)
    len_val_dataset = len(val_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                    args.batch_size, 
                                    shuffle=True, 
                                    num_workers=args.workers, 
                                    drop_last=True)
    val_dataloader = DataLoader(val_dataset, 
                                args.test_batch_size,
                                shuffle=False, 
                                num_workers=args.workers, 
                                drop_last=False)

    # Load Label Embedding

    label_emd_path = os.path.join(args.data_path, 'label_emb.pt')
    label_emb = torch.load(label_emd_path, map_location=args.device).to(torch.float32)


    # Build Model
    clip_model, _ = clip.load(args.clip_path, jit=False)
    for param in clip_model.parameters():
        param.requires_grad = False


    model = CLIPVIT(args, clip_model)
    convert_models_to_fp32(model)
    convert_models_to_fp32(clip_model)
    #del clip_model
    #if torch.cuda.device_count() > 1:
        #model = torch.nn.DataParallel(model)
    model = model.to(args.device)

    # Build Optimizer
    optimizer = build_optimizer(args, model)
    asl_loss = create_loss(args.loss)

    for epoch in range(args.epochs):
        if epoch>1:
            break
        model.train()
        # pretrained_dict = torch.load('/data2/liubeiyan/weight/model_epoch_5.pth')
        # pretrained_dict = torch.load('/home/liubeiyan/MKT-trans/logger/first_stage/0 256/model_epoch_5.pth')
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)


        #train(model,  args, optimizer, train_dataloader, logger, label_emb, asl_loss,epoch)
        model.eval()
        test(model, args, val_dataloader, logger, label_emb ,len_val_dataset, epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",                   type=int,   default=42  )
    parser.add_argument("--record_path",            type=str,   default='/home/liubeiyan/logger/')

    parser.add_argument("--clip-path",              type=str,   default='/home/liubeiyan/ViT-B-16.pt')
    parser.add_argument("--data-path",              type=str,   default='/data/liubeiyan/nus-wide/')
    
    parser.add_argument("--batch-size",             type=int,   default=32,     )
    parser.add_argument("--test-batch-size",        type=int,   default=200,    )
    parser.add_argument("--epochs",                 type=int,   default=10,     )
    parser.add_argument("--warmup_epochs",          type=int,   default=2,      )
    parser.add_argument("--lr",                     type=float, default=1e-5,   )
    parser.add_argument("--min_lr",                 type=float, default=1e-7,   )
    parser.add_argument("--weight_decay",           type=float, default=0.05,   )
    parser.add_argument("--workers",                type=int,   default=1,      )
    parser.add_argument("--momentum",               type=float, default=0.95,   )

    parser.add_argument("--input_size",             type=int,   default=224     )
    
    parser.add_argument("--layer_decay",            type=float, default=0.65    )
    parser.add_argument("--fix_layer",              type=int,   default=10      )
    parser.add_argument("--topk",                   type=int,   default=64      )
    parser.add_argument("--clusters",                   type=int,   default=256      )
    parser.add_argument("--select",                   type=int,   default=64      )
    parser.add_argument("--trans_layer",                   type=int,   default=12      )
    parser.add_argument('--loss', default='asl', type=str, help='(mlsm,bce,focal,asl,halfasl,TwoWayLoss)')
    parser.add_argument("--divices",                   type=int,   default=1   )
    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    torch.cuda.set_device(args.divices)
    main(args)
    
