from collections import OrderedDict

import torch
import torch.nn as nn
from .timm_models.util.layers import DropPath, Mlp
from .timm_models.vision_transformer import *
import math
import torch.nn.functional as F

class CLIPVIT(nn.Module):

    def __init__(self, args, clip_model,embed_dim=768):
        super().__init__()

        self.final_dim = 512
        self.global_only = False
        self.linear=nn.Linear(512,768)        
        self.linear1=nn.Linear(768,512)

        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.clipzero = False

        self.use_clip_proj = False

        if not self.use_clip_proj:
            self.projection = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(embed_dim, self.final_dim)),
                ('act', nn.Tanh()),
                ('fc2', nn.Linear(self.final_dim, self.final_dim))], )
            )

        self.topk = args.topk
        self.projection_dist = deepcopy(clip_model.visual.proj)
        self.trans_layer = args.trans_layer

        self.positional_embedding1 = nn.Parameter(torch.empty(197+args.clusters, 768))
        #self.positional_embedding1 = nn.Parameter(torch.empty(args.clusters, 768))
        nn.init.normal_(self.positional_embedding1, std=0.01)
        self.transformer1 = deepcopy(clip_model.visual.transformer)
        for param in self.transformer1.parameters():
            param.requires_grad = True
        self.projection_dist.requires_grad = True
        self.layernorm=nn.LayerNorm(normalized_shape=768)
        self.layernorm1=nn.LayerNorm(normalized_shape=768)

        self.attention_weights = None
        self.hook_handle = self.transformer1.resblocks[-1].attn.register_forward_hook(self.get_attention_weights)

    def get_attention_weights(self, module, input, output):
        self.attention_weights = output


    def forward_features(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_text(self, x):

        #x = x + self.positional_embedding1
        x = self.layernorm1(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        #x = self.transformer1(x)
        for i in range(self.trans_layer):
            x = self.transformer1.resblocks[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.layernorm(x)
        return x[:,:197,:],x[:,197:,:]



    def forward(self, x, label_embed):

        batch_size=x.shape[0]
        vision_feats = self.forward_features(x)

        tfeat = torch.stack([label_embed for i in range(batch_size)], dim=0)
        tfeat=self.linear(tfeat)
        concatenated_tensor = torch.cat((vision_feats, tfeat), dim=1)
        image,text=self.encode_text(concatenated_tensor)
        final_text=text

        final_image=image@ self.projection_dist
        final_text=final_text@ self.projection_dist



        final_text = final_text / final_text.norm(dim=-1, keepdim=True)
        final_image = final_image[:,0,:] / final_image[:,0,:].norm(dim=1, keepdim=True)
        text_features_conv = final_text .unsqueeze(-1) 
        image_features_conv = final_image.unsqueeze(-1)


        return image_features_conv,text_features_conv














class CLIPVIT0(nn.Module):

    def __init__(self, args, clip_model,embed_dim=768):
        super().__init__()

        self.final_dim = 512
        self.global_only = False
        self.linear=nn.Linear(512,768)        
        self.linear1=nn.Linear(768,512)

        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.clipzero = False

        self.use_clip_proj = False

        if not self.use_clip_proj:
            self.projection = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(embed_dim, self.final_dim)),
                ('act', nn.Tanh()),
                ('fc2', nn.Linear(self.final_dim, self.final_dim))], )
            )

        self.topk = args.topk
        self.projection_dist = deepcopy(clip_model.visual.proj)
        self.trans_layer = args.trans_layer

        self.positional_embedding1 = nn.Parameter(torch.empty(197+args.clusters, 768))
        #self.positional_embedding1 = nn.Parameter(torch.empty(args.clusters, 768))
        nn.init.normal_(self.positional_embedding1, std=0.01)
        self.transformer1 = deepcopy(clip_model.visual.transformer)
        for param in self.transformer1.parameters():
            param.requires_grad = True
        self.projection_dist.requires_grad = True
        self.layernorm=nn.LayerNorm(normalized_shape=768)
        self.layernorm1=nn.LayerNorm(normalized_shape=768)

        self.attention_weights = None
        self.hook_handle = self.transformer1.resblocks[-1].attn.register_forward_hook(self.get_attention_weights)

    def get_attention_weights(self, module, input, output):
        self.attention_weights = output


    def forward_features(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_text(self, x):

        #x = x + self.positional_embedding1
        x = self.layernorm1(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        #x = self.transformer1(x)
        for i in range(self.trans_layer):
            x = self.transformer1.resblocks[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.layernorm(x)
        return x[:,:197,:],x[:,197:,:]



    def forward(self, x, label_embed):

        batch_size=x.shape[0]
        vision_feats = self.forward_features(x)

        tfeat = torch.stack([label_embed for i in range(batch_size)], dim=0)
        tfeat=self.linear(tfeat)
        concatenated_tensor = torch.cat((vision_feats, tfeat), dim=1)
        image,text=self.encode_text(concatenated_tensor)
        #attention=torch.diagonal(self.attention_weights[1],dim1=1,dim2=2)
        #print(attention[:,197:])
        #mmm=F.softmax(attention[:,1:]*(197+args.clusters), dim=-1)
        #print(mmm)
        final_text=text

        final_image=image@ self.projection_dist

        final_text=final_text@ self.projection_dist


        #print(final_text.shape)
        return final_image[:,1:,:],final_image[:,0,:],final_text
        #return final_image[:,0,:]
        #return final_text

