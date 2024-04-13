import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version

from .glimpse_transformer import build_glimpse_transformer
import torchvision
import math

from mmdet.models.backbones.resnet import BasicBlock
import pdb

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes
def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def denorm_2d_pts_bev(pts, bev_h, bew_w):
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] * bev_w
    new_pts[...,1:2] = pts[...,1:2] * bev_h
    return new_pts
     
class ProcessNet(nn.Module):
    
    def __init__(self, in_channels, out_channels,act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        #self.conv_expand = nn.Conv2d(in_channels, in_channels // 2,  3, padding=1, bias=True)
        #self.conv_basicblock = BasicBlock(in_channels // 2, in_channels // 2)
        self.conv_reduce = nn.Conv2d(in_channels , out_channels, 1, bias=True)

    def forward(self, x_se):
        #x_se = self.conv_expand(x_se)
        #x_se = self.conv_basicblock(x_se)
        x_se = self.conv_reduce(x_se)
        return x_se
    

class ChangeChannels(nn.Module):
    
    def __init__(self, in_channels=2, out_channels=32, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_expand = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.conv_basicblock = BasicBlock(out_channels, out_channels)

    def forward(self, x_se):
        x_se = self.conv_expand(x_se)
        x_se = self.conv_basicblock(x_se)
        return x_se
        

@HEADS.register_module()
class MapHead_Ins(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_vec=20,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 random_refpoints_xy=True,
                 query_embed_type='all_pts',
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 use_rego=False,
                 loss_pts=dict(type='ChamferDistance', 
                             loss_src_weight=1.0, 
                             loss_dst_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_fn = dict(type='SimpleLoss',pos_weight=2.13, loss_weight=3), 
                 loss_dice = dict(type='DiceLoss', loss_weight=15),
                 loss_fn_ = dict(type='SimpleLoss',pos_weight=2.13, loss_weight=3), 
                 loss_dice_ = dict(type='DiceLoss', loss_weight=15),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.random_refpoints_xy = random_refpoints_xy
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.counter=0
        
        self.use_rego=use_rego
        super(MapHead_Ins, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)
        self.loss_fn = build_loss(loss_fn)
        self.loss_dice = build_loss(loss_dice)
        self.loss_fn_ = build_loss(loss_fn_)
        self.loss_dice_ = build_loss(loss_dice_)

        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec

        self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.conv_seg = BasicConv2d(128, 256, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(128, 2, 1)
        self.process_net = ProcessNet(int(self.embed_dims +32+ 2), self.embed_dims)
        self.change_channels = ChangeChannels(2, 32)
        if self.use_rego:
            self.rego_scales = [1.0, 1.0]
            self.dropout = nn.Dropout(p=0.01)
            self.roi_query_dim = 256
            self.roi_feat_dim =self.roi_query_dim
            self.feat_gp = 4
            self.roi_ext = torchvision.ops.MultiScaleRoIAlign(['feat1','feat2','feat3'],5,2) 
     
            hidden_dim = 256

#   

#   

            self.glimpse_transformer = build_glimpse_transformer() #if args.use_rego else None
            num_pred = (self.transformer.glimpse_decoder.num_layers + 1) if \
                self.as_two_stage else self.transformer.decoder.num_layers
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            for gi in range(len(self.rego_scales)):
                rcnn_net = nn.Sequential( *[nn.Conv2d(256, self.roi_feat_dim, kernel_size=5, stride=1, padding=0, groups=self.feat_gp),   #
                                                nn.Flatten(1), nn.LayerNorm(self.roi_feat_dim), nn.ReLU(),  
                                                nn.Linear(self.roi_feat_dim, self.roi_query_dim), nn.LayerNorm(self.roi_query_dim) ])
                setattr(self, 'rcnn_net_%d'%gi, rcnn_net)
                rego_hs_linear = nn.Linear((gi+1) * hidden_dim, hidden_dim, bias=False)
                rego_hs_linear_norm = nn.LayerNorm(hidden_dim)
                setattr(self, 'rego_hs_linear_%d'%gi, rego_hs_linear)
                setattr(self, 'rego_hs_linear_norm_%d'%gi, rego_hs_linear_norm)
                if gi == 0:
                    setattr(self, 'glimpse_transformer_%d'%gi, self.glimpse_transformer)
                else:
                    setattr(self, 'glimpse_transformer_%d'%gi, copy.deepcopy(self.glimpse_transformer))
#   
                rego_hs_fuser = nn.Linear((gi+2) * 256, 256, bias=False)
                setattr(self, 'rego_hs_fuser_%d'%gi, _get_clones(rego_hs_fuser, num_pred))
                setattr(self, 'layer_norms_%d'%gi, nn.ModuleList([nn.LayerNorm(hidden_dim) for i in range(num_pred)]))
#   
#   
                for m_str in ['rcnn_net', 'layer_norms']: 
                    m = getattr(self, m_str + '_%d'%gi)
                    for mm in m.modules():
                        if isinstance(mm, nn.Conv2d):
                            nn.init.xavier_normal_(mm.weight)
                            nn.init.constant_(mm.bias, 0.0)
                        elif isinstance(mm, nn.Linear):
                            nn.init.xavier_normal_(mm.weight)
                            nn.init.constant_(mm.bias, 0.0)
                        elif isinstance(mm, nn.LayerNorm):
                            nn.init.constant_(mm.weight, 1.0)
                            nn.init.constant_(mm.bias, 0.0)
                m = getattr(self, 'rego_hs_fuser_%d'%gi)
                for mm in m.modules():
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_normal_(mm.weight)
#
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        ##add new here


        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers+4

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            
            #self.valid_branches = nn.ModuleList(
            #    [fc_valid for _ in range(num_pred)])

            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            if self.bev_encoder_type == 'BEVFormerEncoder':
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding_ = nn.Linear(self.embed_dims*2, self.num_pts_per_vec*self.embed_dims*2)
                #self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    
    def gen_grid_2d(self, H, W, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y[None] / H
        ref_x = ref_x[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1, 1).permute(0, 3, 1, 2)
        return ref_2d
        
        
    @force_fp32(apply_to=('multi_feats', 'prev_bev'))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None,  only_bev=False, **kwargs):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        #
        if lidar_feat!=None:
          bs = lidar_feat.shape[0]
          dtype = lidar_feat.dtype
        else:
          bs, num_cam, _, _, _ = mlvl_feats[0].shape
          dtype = mlvl_feats[0].dtype

        #pdb.set_trace()
        # import pdb;pdb.set_trace()
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding_(self.instance_embedding.weight).reshape(self.num_vec,self.num_pts_per_vec,-1)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = ( pts_embeds + instance_embeds).flatten(0, 1).to(dtype)

        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None
        #only_bev=False
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            bev_embed = self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            bs=mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, self.bev_h, self.bev_w, -1).permute(0,3,1,2).contiguous()
            #add sad
            # lidar_feat, multi_bev_feats,loss_sad = self.bevencode(bev_embed)
            lidar_feat, multi_bev_feats = self.bevencode(bev_embed)
            return lidar_feat
        else:
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w, 
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None, 
                #valid_branches=self.valid_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        #import pdb
        #pdb.set_trace()
        bev_embed, multi_bev_feats, hs, init_reference, inter_references, inter_ins = outputs
        
        
        bev_feats = self.conv_2(bev_embed)
        seg = self.semantic_output(bev_feats)
        

        bev_seg_prob = torch.softmax(seg,dim=1)
        bev_seg_new = self.change_channels(bev_seg_prob)
        grid_2d = torch.abs( (self.gen_grid_2d(H=self.bev_h, W=self.bev_w, bs=bs, device=bev_feats.device, dtype=dtype) - 0.5))
        bev_feature = torch.cat((bev_embed, bev_seg_new, grid_2d), dim=1)
        bev_feature = self.process_net(bev_feature)


        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
         
            # vec_embedding = hs[lvl].reshape(bs, self.num_vec, -1)
            outputs_class = self.cls_branches[lvl](hs[lvl]
                                            .view(bs,self.num_vec, self.num_pts_per_vec,-1)
                                            .mean(2))
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 2 
            tmp[..., 0:2] += reference[..., 0:2]
            # tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp = tmp.sigmoid() # cx,cy,w,h
          
            # TODO: check if using sigmoid
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)


            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord) 
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_pts_preds': outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_pts_preds': None,
            'seg_results': seg,
            'ins_results':inter_ins 
        }

        if self.use_rego:
            #bev_seg_feats = self.conv_seg(bev_feats)
            
            rego_dic ={'feat1':multi_bev_feats[1],'feat2':multi_bev_feats[2],'feat3':bev_feature}
          
            prev_dec_hs = hs[-1]#2 1k 256
            prev_coord = outputs_coord[-1].detach()
            prev_pts_coord = outputs_pts_coords[-1].detach()  #bs 50 20 2

            
            outputs_pts_coords_ = []
            #outputs_valids_=[]
            outputs_coords_ = []
            outputs_classes_ = []
            outputs_ins_ = []
            for gi in range(len(self.rego_scales)):
                prev_coord = torch.zeros((bs, prev_pts_coord.shape[1], prev_pts_coord.shape[2],4 ), device=prev_pts_coord.device)
                prev_coord[...,:2] = prev_pts_coord[...,:]
                prev_coord[...,2] = 0.1
                prev_coord[...,3] = 0.1
                prev_coord = prev_coord.reshape(bs, self.num_query, 4)
                
                with torch.no_grad():
                    feats_shape_tensor = torch.ones( (bs, 1, 4), device=bev_embed.device)
                    feats_shape_tensor[:, 0, 0::2] = self.bev_w #x
                    feats_shape_tensor[:, 0, 1::2] = self.bev_h #y
                    scalar_tensor = torch.ones( (bs, 1, 4), device=bev_embed.device)
                    scalar_tensor[:,:,2:] = self.rego_scales[gi]
                pred_bboxes = (prev_coord * scalar_tensor).clamp(max=1.0)
                pred_bboxes = pred_bboxes * feats_shape_tensor
                pred_bboxes = bbox_cxcywh_to_xyxy(pred_bboxes)
                pred_bboxes = [pred_bboxes[i] for i in range(bs)]
                

              
                ext_roi_feat = self.roi_ext(rego_dic
                , pred_bboxes, [(self.bev_h, self.bev_w)])
###
                ext_roi_feat = getattr(self, 'rcnn_net_%d'%gi)(ext_roi_feat) 
                rego_in = ext_roi_feat.view(bs, -1 , ext_roi_feat.shape[-1] )
                prev_hs = getattr(self, 'rego_hs_linear_%d'%gi)(prev_dec_hs)
                prev_hs = getattr(self, 'rego_hs_linear_norm_%d'%gi)(prev_hs)
###
                #prev_hs = prev_dec_hs
                rego_hs = getattr(self, 'glimpse_transformer_%d'%gi)(rego_in, prev_hs)[0]
###
                rego_output_classes = []

                rego_output_coords = []
                rego_output_pts_coords = []
                reference_reg = inverse_sigmoid(prev_pts_coord)
                hs_fusers = getattr(self, 'rego_hs_fuser_%d'%gi)
                l_norms = getattr(self, 'layer_norms_%d'%gi)
                prev_h = prev_dec_hs.detach() 
                for lvl in range(rego_hs.shape[0]):
                    fuse_h = torch.cat((prev_h, rego_hs[lvl]), 2) 
                    fuse_h = hs_fusers[lvl](fuse_h) 
                    fuse_h = l_norms[lvl](fuse_h)

                    output_class = self.cls_branches[gi+lvl+4](fuse_h.view(bs,self.num_vec, self.num_pts_per_vec,-1).mean(2))
                    reference_reg = reference_reg.view(bs, -1, 2) + self.reg_branches[gi+lvl+4](fuse_h) 
                    tmp = reference_reg.sigmoid()
                    output_coord, output_pts_coord = self.transform_box(tmp)
                    
                    outputs_coords_.append(output_coord)
                    outputs_pts_coords_.append(output_pts_coord)
                    outputs_classes_.append(output_class)


                    pad_ins = torch.zeros_like(inter_ins[-1])
                    outputs_ins_.append(pad_ins)
###
      
###
###
                rego_outputs_classes_ = torch.stack(outputs_classes_)
                rego_outputs_pts_coords_ = torch.stack(outputs_pts_coords_)
                rego_outputs_coords_ = torch.stack(outputs_coords_)
                

                rego_outputs_ins_ = torch.stack(outputs_ins_)
                prev_dec_hs = torch.cat( (prev_dec_hs, rego_hs[-1]), 2)
                prev_pts_coord = rego_outputs_pts_coords_[-1].detach()
            outputs_pts_coords = torch.cat((outputs_pts_coords, rego_outputs_pts_coords_), 0 )
            outputs_coords = torch.cat((outputs_coords, rego_outputs_coords_), 0 )
            outputs_classes = torch.cat((outputs_classes, rego_outputs_classes_), 0)

            outputs_ins = torch.cat((inter_ins, rego_outputs_ins_), 0)
            outs['all_pts_preds'] = outputs_pts_coords
            outs['all_cls_scores'] = outputs_classes
            outs['all_bbox_preds'] = outputs_coords
            outs['ins_results'] = outputs_ins
        return outs
        

    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.num_vec,
                                self.num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
       

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           ins_pred,
                           gt_inses,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
       
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        #import pdb; pdb.set_trace()
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,#ins_pred,gt_inses,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR 
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]

        gt_ordered_inses = torch.zeros_like(gt_inses)
        gt_ordered_inses[pos_inds,:,:]=gt_inses[sampling_result.pos_assigned_gt_inds,:,:]
        ins_weights = torch.zeros(gt_inses.shape[0]).to(label_weights.device)
        ins_weights[pos_inds]=1.0
        #ins_pred=ins_pred[pos_inds]
        #gt_inses=gt_inses[sampling_result.pos_assigned_gt_inds]
        #ins_weights = torch.ones(gt_inses.shape[0]).to(label_weights.device)
        return (labels, label_weights, bbox_targets, bbox_weights,gt_ordered_inses, ins_pred,ins_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    ins_preds_list,
                    gt_ins_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs) 
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, gt_ins_list, ins_pred_list, ins_weights_list,pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,ins_preds_list, gt_ins_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,gt_ins_list, ins_pred_list,ins_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    ins_preds,
                    gt_ins_list,
                    #valid_scores,
                    #rego_cls_scores,
                    #rego_pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = bbox_preds.size(0)
        #rego_num = rego_cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        ins_preds_list = [ins_preds[i] for i in range(num_imgs)]




        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list, ins_preds_list, gt_ins_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,gt_ins_list, ins_preds_list,ins_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        ins_preds =torch.cat(ins_preds_list,0)
        ins_gt = torch.cat(gt_ins_list,0)
        ins_weights = torch.cat(ins_weights_list,0)
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
            #valid_avg_factor = reduce_mean(valid_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        #valid_avg_factor = max(valid_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                                               :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
        #valid_scores = valid_scores.reshape(-1, valid_scores.size(-2), valid_scores.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan,
                                                            :,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()

        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan,
                                                                          :,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

       
        loss_ins= self.loss_fn_(ins_preds, ins_gt,ins_weights)+self.loss_dice_(ins_preds, ins_gt,ins_weights)
       

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
            loss_ins = torch.nan_to_num(loss_ins)
        #rego_loss_cls, rego_loss_pts, rego_loss_dir=None, None, None    
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir, loss_ins

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             seg_gt,
             ins_gt,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        # import pdb;pdb.set_trace()
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds  = preds_dicts['all_pts_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        enc_pts_preds  = preds_dicts['enc_pts_preds']

        seg = preds_dicts['seg_results']
        all_ins = preds_dicts['ins_results']
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device


        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        if self.gt_shift_pts_pattern == 'v0':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v1':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v2':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v3':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v4':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
        else:
            raise NotImplementedError
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        all_seg_labels_list = [seg_gt for _ in range(num_dec_layers)]
        all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_ins_list = [ins_gt for _ in range(num_dec_layers)]
        # import pdb;pdb.set_trace()
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir, losses_ins = multi_apply( 
            self.loss_single, all_cls_scores, all_bbox_preds,all_pts_preds, all_ins, all_gt_ins_list,
            all_gt_bboxes_list, all_gt_labels_list,all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list)
        seg_gt = torch.stack(seg_gt)
        ins_gt = torch.stack(ins_gt)
        seg_weight = torch.ones(seg_gt.shape[0]).to(seg_gt.device)
        losses_seg = self.loss_fn(seg, seg_gt,seg_weight) + self.loss_dice(seg, seg_gt,seg_weight)
        losses_seg = torch.nan_to_num(losses_seg)
##
      
        losses_ins = losses_ins[:6]
        loss_dict = dict()
        
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            # TODO bug here
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_losses_pts, enc_losses_dir = self.loss_single(
                enc_cls_scores, enc_bbox_preds, enc_pts_preds,
                gt_bboxes_list, binary_labels_list, gt_pts_list,gt_bboxes_ignore)

            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_losses_iou'] = enc_losses_iou
            loss_dict['enc_losses_pts'] = enc_losses_pts
            loss_dict['enc_losses_dir'] = enc_losses_dir

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        #loss_dict['loss_valid'] = losses_valid[-1]
        loss_dict['loss_seg'] = losses_seg

        loss_dict['loss_ins'] = losses_ins[-1]

      
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1],
                                           losses_iou[:-1],
                                           losses_pts[:-1],
                                           losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            
            num_dec_layer += 1

        num_dec_layer = 0
        for loss_ins_i in losses_ins[:-1]:
            loss_dict[f'd{num_dec_layer}.loss_ins'] = loss_ins_i
            num_dec_layer += 1

        self.counter+=1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts) 
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']
            #valid_labels = preds['valid_labels']
            ret_list.append([bboxes, scores, labels, pts])

        return ret_list