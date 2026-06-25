import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
import utils.lr_sched as lrs
from models.rank_loss import ranking_lossT
from utils.misc import compute_F1, compute_AP
from sklearn.cluster import KMeans
#import cupy as cp
#from cuml.cluster import KMeans
from utils.losses import create_loss
import numpy as np


def train(model,  args, optimizer, dataloader, logger, label_emb,asl_loss,epoch):

    logger.info("\nTRAINING MODE")

    mean_rank_loss = 0

    label_embed = label_emb.cpu()
    label_embed=    label_embed.numpy()
    kmeans = KMeans(n_clusters=args.clusters,n_init=30)
    kmeans.fit(label_embed)
    label_embed = torch.tensor(kmeans.cluster_centers_).to(torch.float32).to(args.device)

    for i, (train_inputs, train_labels,path) in enumerate(tqdm(dataloader)):
        #if i>2:
            #break

        lrs.adjust_learning_rate(optimizer, i / len(dataloader) + epoch, args)

        # import pdb; pdb.set_trace() print(torch.nonzero(1+train_labels[2]))
        optimizer.zero_grad()
        
        ### remove empty label images while training ###
        temp_label = torch.clamp(train_labels,0,1)
        temp_seen_labels = temp_label.sum(1)
        temp_label = temp_label[temp_seen_labels>0]
        train_labels   = train_labels[temp_seen_labels>0]
        train_inputs   = train_inputs[temp_seen_labels>0]

        train_inputs = train_inputs.cuda()
        train_labels = train_labels.cuda()
        
        local,globally,text = model(train_inputs, label_embed)
        score1 = torch.topk(local @ label_emb[:925].t(), k=args.topk, dim=1)[0].mean(dim=1)
        score3 = torch.topk(text @ label_emb[:925].t(), k=args.select, dim=1)[0].mean(dim=1)
        score2 =globally.squeeze(dim=1) @ label_emb[:925].t()

        logits =( score2 + score3+score1)/3


        train_labels=torch.clamp(train_labels, min=0)
        rank_loss = asl_loss(logits, train_labels.float())

        loss =  rank_loss

        mean_rank_loss += rank_loss.item()

        loss.requires_grad_()
        loss.backward()
        
        optimizer.step()

    mean_rank_loss /= len(dataloader)

    learning_rate = optimizer.param_groups[-1]['lr']
    
    logger.info("------------------------------------------------------------------")
    logger.info("FINETUNING Epoch: {}/{} \tRankLoss: {:.6f}\tLearningRate {}".format(epoch, args.epochs, mean_rank_loss, learning_rate))
    logger.info("------------------------------------------------------------------")

    torch.save(model.state_dict(), os.path.join(args.record_path, "model_epoch_{}.pth".format(epoch)))


########### TEST FUNC ###########
def test(model, args, dataloader, logger, label_emb, len_testdataset, writer, epoch=-1):

    logger.info("\n=======================EVALUATION MODE=======================")
    

    prediction_81 = torch.empty(len_testdataset,81)
    prediction_1006 = torch.empty(len_testdataset,1006)
    lab_81 = torch.empty(len_testdataset,81)
    lab_1006 = torch.empty(len_testdataset,1006)
    
    test_batch_size = args.test_batch_size

    label_embed = label_emb.cpu()
    label_embed=    label_embed.numpy()

    kmeans = KMeans(n_clusters=args.clusters,n_init=30)
    kmeans.fit(label_embed)
    label_embed = torch.tensor(kmeans.cluster_centers_).to(torch.float32).to(args.device)
    cnt = 0
    for features, labels_1006, labels_81, _ in tqdm(dataloader):
        strt = cnt
        endt = min(cnt + test_batch_size, len_testdataset)
        cnt += test_batch_size
        #if cnt>300:
            #break
        with torch.no_grad():
            
            local,globally ,text= model(features.cuda(),label_embed)
            score1 = torch.topk(local @ label_emb[925:].t(),k=model.topk, dim=1)[0].mean(dim=1)
            score3 = torch.topk(text @ label_emb[925:].t(),k=args.select, dim=1)[0].mean(dim=1)
            score2 = globally.squeeze(dim=1) @ label_emb[925:].t()
            logits_81 = ( score2 +score3+ score1)/3

            score1 = torch.topk(local @ label_emb.t(),k=model.topk, dim=1)[0].mean(dim=1)
            score3 = torch.topk(text @ label_emb.t(),k=args.select, dim=1)[0].mean(dim=1)
            score2 = globally.squeeze(dim=1) @ label_emb.t()
            logits_1006 = ( score2 +score3+score1)/3

        
        prediction_81[strt:endt,:] = logits_81
        prediction_1006[strt:endt,:] = logits_1006
        lab_81[strt:endt,:] = labels_81
        lab_1006[strt:endt,:] = labels_1006
    
    logger.info("completed calculating predictions over all images")
    logits_81_5 = prediction_81.clone()
    print(prediction_81.shape)
    print(lab_81.shape)
    ap_81 = compute_AP(prediction_81.cuda(), lab_81.cuda())

    F1_3_81,P_3_81,R_3_81 = compute_F1(prediction_81.cuda(), lab_81.cuda(), 'overall', k_val=3)
    F1_5_81,P_5_81,R_5_81 = compute_F1(logits_81_5.cuda(), lab_81.cuda(), 'overall', k_val=5)

    logger.info('ZSL AP: %.4f',torch.mean(ap_81))
    logger.info('k=3: %.4f,%.4f,%.4f',torch.mean(F1_3_81),torch.mean(P_3_81),torch.mean(R_3_81))
    logger.info('k=5: %.4f,%.4f,%.4f',torch.mean(F1_5_81),torch.mean(P_5_81),torch.mean(R_5_81))

    logits_1006_5 = prediction_1006.clone()
    #print(lab_1006[0].shape)
    ap_1006 = compute_AP(prediction_1006.cuda(), lab_1006.cuda())
    F1_3_1006,P_3_1006,R_3_1006 = compute_F1(prediction_1006.cuda(), lab_1006.cuda(), 'overall', k_val=3)
    F1_5_1006,P_5_1006,R_5_1006 = compute_F1(logits_1006_5.cuda(), lab_1006.cuda(), 'overall', k_val=5)

    logger.info('GZSL AP:%.4f',torch.mean(ap_1006))
    logger.info('g_k=3:%.4f,%.4f,%.4f',torch.mean(F1_3_1006), torch.mean(P_3_1006), torch.mean(R_3_1006))
    logger.info('g_k=5:%.4f,%.4f,%.4f',torch.mean(F1_5_1006), torch.mean(P_5_1006), torch.mean(R_5_1006))

