import math
from functools import partial
import logging
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from .longformer2d import Long2DSCSelfAttention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 rpe=False, wx=14, wy=14, nglo=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Inspired by swin transformer:
        # https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L88-L103
        # define parameter tables for local and global relative position bias
        self.rpe = rpe
        if rpe:
            self.wx = wx
            self.wy = wy
            self.nglo = nglo
            self.local_relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * wx - 1) * (2 * wy - 1),
                            num_heads))  # (2*wx-1, 2*wy-1, nH)
            trunc_normal_(self.local_relative_position_bias_table, std=.02)
            if nglo >= 1:
                self.g2l_relative_position_bias = nn.Parameter(
                    torch.zeros(2, num_heads, nglo))  # (2, nH, nglo)
                self.g2g_relative_position_bias = nn.Parameter(
                    torch.zeros(num_heads, nglo, nglo))  # (nH, nglo, nglo)
                trunc_normal_(self.g2l_relative_position_bias, std=.02)
                trunc_normal_(self.g2g_relative_position_bias, std=.02)

            # get pair-wise relative position index
            coords_h = torch.arange(wx)
            coords_w = torch.arange(wy)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wx, wy
            coords_flatten = torch.flatten(coords, 1)  # 2, Wx*Wy
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wx*Wy, Wx*Wy
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wx*Wy, Wx*Wy, 2
            relative_coords[:, :, 0] += wx - 1  # shift to start from 0
            relative_coords[:, :, 1] += wy - 1
            relative_coords[:, :, 0] *= 2 * wy - 1
            relative_position_index = relative_coords.sum(-1)  # Wx*Wy, Wx*Wy
            self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rpe:
            assert N == self.nglo + self.wx*self.wy, "For relative position, N != self.nglo + self.wx*self.wy!"
            local_relative_position_bias = self.local_relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                self.wx*self.wy, self.wx*self.wy, -1)  # Wh*Ww, Wh*Ww,nH
            relative_position_bias = local_relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            if self.nglo > 0:
                # relative position embedding of global tokens
                global_relative_position_bias = torch.cat([
                    self.g2g_relative_position_bias,
                    self.g2l_relative_position_bias[0].unsqueeze(-1).expand(-1, -1, self.wx*self.wy)
                ], dim=-1)  # nH, nglo, N
                # relative position embedding of local tokens
                local_relative_position_bias = torch.cat([
                    self.g2l_relative_position_bias[1].unsqueeze(1).expand(-1, self.wx*self.wy, -1),
                    relative_position_bias,
                ], dim=-1)  # nH, Wh*Ww, N
                relative_position_bias = torch.cat([
                    global_relative_position_bias,
                    local_relative_position_bias,
                ], dim=1)  # nH, N, N
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = (attn - torch.max(attn, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        S = T
        macs = 0
        n_params = 0

        # Scaled-dot-product macs
        # [B x T x C] x [B x C x S] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        num_macs_kq = T * S * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = T * C * S

        macs += num_macs_kq + num_macs_v
        # print('macs att', macs / 1e8)

        # self attention: T should be equal to S
        assert T == S
        qkv_params = sum([p.numel() for p in module.qkv.parameters()])
        n_params += qkv_params
        # multiply by Seq length
        macs += qkv_params * T
        # print('macs qkv', qkv_params * T / 1e8)

        proj_params = sum([p.numel() for p in module.proj.parameters()])
        n_params += proj_params
        macs += (proj_params * T)
        # print('macs proj', proj_params * T / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size, nx, ny, in_chans=3, embed_dim=768, nglo=1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_embed=True,
                 drop_rate=0.0, ape=True):
        # maximal global/x-direction/y-direction tokens: nglo, nx, ny
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

        self.norm_embed = norm_layer(embed_dim) if norm_embed else None

        self.nx = nx
        self.ny = ny
        self.Nglo = nglo
        if nglo >= 1:
            self.cls_token = nn.Parameter(torch.zeros(1, nglo, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None
        self.ape = ape
        if ape:
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, nglo, embed_dim))
            self.x_pos_embed = nn.Parameter(torch.zeros(1, nx, embed_dim // 2))
            self.y_pos_embed = nn.Parameter(torch.zeros(1, ny, embed_dim // 2))
            trunc_normal_(self.cls_pos_embed, std=.02)
            trunc_normal_(self.x_pos_embed, std=.02)
            trunc_normal_(self.y_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, xtuple):
        x, nx, ny = xtuple
        B = x.shape[0]

        x = self.proj(x)
        nx, ny = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        assert nx <= self.nx and ny <= self.ny, "Input size {} {} should <= {} {}!".format(nx, ny, self.nx, self.ny)

        if self.norm_embed:
            x = self.norm_embed(x)

        # concat cls_token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.ape:
            # add position embedding
            i = torch.arange(nx, device=x.device)
            j = torch.arange(ny, device=x.device)
            x_emb = self.x_pos_embed[:, i, :]
            y_emb = self.y_pos_embed[:, j, :]
            pos_embed_2d = torch.cat([
                x_emb.unsqueeze(2).expand(-1, -1, ny, -1),
                y_emb.unsqueeze(1).expand(-1, nx, -1, -1),
            ], dim=-1).flatten(start_dim=1, end_dim=2)
            x = x + torch.cat([self.cls_pos_embed, pos_embed_2d], dim=1).expand(
                B, -1, -1)

        x = self.pos_drop(x)

        return x, nx, ny


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# for Performer, start
def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# for Performer, end


class AttnBlock(nn.Module):
    """ Meta Attn Block
    """
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 attn_type='full', w=7, d=1, sharew=False, nglo=1,
                 only_glo=False,
                 seq_len=None, num_feats=256, share_kv=False, sw_exact=0,
                 rratio=2, rpe=False, wx=14, wy=14,
                 mode=0):
        super().__init__()
        self.norm = norm_layer(dim)
        if attn_type == 'full':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop,
                                  proj_drop=drop,
                                  rpe=rpe, wx=wx, wy=wy, nglo=nglo)
        elif attn_type == 'longformerhand':
            self.attn = Long2DSCSelfAttention(
                dim, exact=sw_exact, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, w=w, d=d, sharew=sharew,
                nglo=nglo, only_glo=only_glo, autograd=False,
                rpe=rpe, mode=mode
            )
        elif attn_type == 'longformerauto':
            self.attn = Long2DSCSelfAttention(
                dim, exact=sw_exact, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, w=w, d=d, sharew=sharew,
                nglo=nglo, only_glo=only_glo, autograd=True,
                rpe=rpe, mode=mode
            )
        else:
            raise ValueError(
                "Not supported attention type {}".format(attn_type))
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, xtuple):
        x, nx, ny = xtuple
        x = x + self.drop_path(self.attn(self.norm(x), nx, ny))
        return x, nx, ny


class MlpBlock(nn.Module):
    """ Meta MLP Block
    """

    def __init__(self, dim, out_dim=None, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=out_dim, act_layer=act_layer, drop=drop)
        self.shortcut = nn.Identity()
        if out_dim is not None and out_dim != dim:
            self.shortcut = nn.Sequential(nn.Linear(dim, out_dim),
                                          nn.Dropout(drop))

    def forward(self, xtuple):
        x, nx, ny = xtuple
        x = self.shortcut(x) + self.drop_path(self.mlp(self.norm(x)))
        return x, nx, ny


def parse_arch(layer_cfgstr):
    layer_cfg = {'l': 1, 'h': 3, 'd': 192, 'n': 1, 's': 1, 'g': 1,
                     'p': 2, 'f': 7, 'a': 0}  # defaults
    for attr in layer_cfgstr.split(','):
        layer_cfg[attr[0]] = int(attr[1:])
    return layer_cfg


class MsViT(nn.Module):
    """ Multiscale Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, arch, img_size=512, in_chans=3,
                 num_classes=1000,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_embed=False, w=7, d=1, sharew=False, only_glo=False,
                 share_kv=False,
                 attn_type='longformerhand', sw_exact=0, mode=0,
                 out_features=None,
                 freeze_at=0, #detectron2
                 **args):
        super().__init__()
        self.num_classes = num_classes

        if 'ln_eps' in args:
            ln_eps = args['ln_eps']
            self.norm_layer = partial(nn.LayerNorm, eps=ln_eps)
            logging.info("Customized LayerNorm EPS: {}".format(ln_eps))
        else:
            self.norm_layer = norm_layer
        self.drop_path_rate = drop_path_rate
        self.attn_type = attn_type

        self.attn_args = dict({
            'attn_type': attn_type,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'drop': drop_rate,
            'attn_drop': attn_drop_rate,
            'w': w,
            'd': d,
            'sharew': sharew,
            'only_glo': only_glo,
            'share_kv': share_kv,
            'sw_exact': sw_exact,
            'norm_layer': norm_layer,
            'mode': mode,
        })
        self.patch_embed_args = dict({
            'norm_layer': norm_layer,
            'norm_embed': norm_embed,
            'drop_rate': drop_rate,
        })
        self.mlp_args = dict({
            'mlp_ratio': 4.0,
            'norm_layer': norm_layer,
            'act_layer': nn.GELU,
            'drop': drop_rate,
        })

        # Attributes for maskrcnn
        assert out_features, "out_features is empty!"
        self._out_feature_strides = []
        self._out_feature_channels = []
        self._out_features = out_features
        self.frozen_stages = freeze_at

        self.layer_cfgs = [parse_arch(layer) for layer in arch.split('_')]
        self.num_layers = len(self.layer_cfgs)
        self.depth = sum([cfg['n'] for cfg in self.layer_cfgs])
        self.out_planes = self.layer_cfgs[-1]['d']
        self.Nglos = [cfg['g'] for cfg in self.layer_cfgs]
        self.avg_pool = args['avg_pool'] if 'avg_pool' in args else False

        # ensure divisibility
        stride = 1
        down_strides = []
        for cfg in self.layer_cfgs:
            stride *= cfg['p']
            down_strides.append(stride)
        self._size_divisibility = stride
        self.Nx = (img_size + (stride - 1)) // stride * stride
        self.Ny = (img_size + (stride - 1)) // stride * stride

        dprs = torch.linspace(0, drop_path_rate, self.depth).split(
            [cfg['n'] for cfg in self.layer_cfgs]
        )  # stochastic depth decay rule
        self.layer1 = self._make_layer(in_chans, self.layer_cfgs[0],
                                       dprs=dprs[0], layerid=1)
        if "layer1" in self._out_features:
            self._out_feature_strides.append(down_strides[0])
            self._out_feature_channels.append(self.layer_cfgs[0]['d'])

        self.layer2 = self._make_layer(self.layer_cfgs[0]['d'],
                                       self.layer_cfgs[1], dprs=dprs[1],
                                       layerid=2)
        if "layer2" in self._out_features:
            self._out_feature_strides.append(down_strides[1])
            self._out_feature_channels.append(self.layer_cfgs[1]['d'])

        self.layer3 = self._make_layer(self.layer_cfgs[1]['d'],
                                       self.layer_cfgs[2], dprs=dprs[2],
                                       layerid=3)
        if "layer3" in self._out_features:
            self._out_feature_strides.append(down_strides[2])
            self._out_feature_channels.append(self.layer_cfgs[2]['d'])

        if self.num_layers == 3:
            self.layer4 = None
        elif self.num_layers == 4:
            self.layer4 = self._make_layer(self.layer_cfgs[2]['d'],
                                           self.layer_cfgs[3], dprs=dprs[3],
                                           layerid=4)
            if "layer4" in self._out_features:
                self._out_feature_strides.append(down_strides[3])
                self._out_feature_channels.append(self.layer_cfgs[3]['d'])
        else:
            raise ValueError("Numer of layers {} not implemented yet!".format(self.num_layers))

        assert self._size_divisibility==stride, "Some stride down layer has been ignored!"

        self.apply(self._init_weights)

    def _freeze_stages(self):
        if self.frozen_stages <= 0:
            return

        if self.frozen_stages >= 1:
            # froze the first patch embeding layer
            self.layer1[0].eval()
            for param in self.layer1[0].parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            # froze layer1 to layer{frozen_stages-1}
            for i in range(1, self.frozen_stages):
                m = getattr(self, "layer" + str(i))
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MsViT, self).train(mode)
        self._freeze_stages()

    def reset_vil_mode(self, mode):
        longformer_attentions = find_modules(self, Long2DSCSelfAttention)
        for longformer_attention in longformer_attentions:
            mode_old = longformer_attention.mode
            if mode_old != mode:
                longformer_attention.mode = mode
                logging.info(
                    "Change vil attention mode from {} to {} in " "layer {}"
                        .format(mode_old, mode, longformer_attention))
        return

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def _make_layer(self, in_dim, layer_cfg, dprs, layerid=0):
        layer_id, num_heads, dim, num_block, is_sparse_attn, nglo, patch_size, num_feats, ape \
            = layer_cfg['l'], layer_cfg['h'], layer_cfg['d'], layer_cfg['n'], \
              layer_cfg['s'], layer_cfg['g'], layer_cfg['p'], layer_cfg['f'], \
              layer_cfg['a']
        assert layerid == layer_id, "Error in _make_layer: layerid {} does not equal to layer_id {}".format(layerid, layer_id)
        self.Nx = nx = self.Nx // patch_size
        self.Ny = ny = self.Ny // patch_size
        seq_len = nx * ny + nglo

        self.attn_args['nglo'] = nglo
        self.patch_embed_args['nglo'] = nglo
        self.attn_args['num_feats'] = num_feats  # shared for linformer and performer
        self.attn_args['rratio'] = num_feats  # srformer reuses this parameter
        self.attn_args['w'] = num_feats  # longformer reuses this parameter
        if is_sparse_attn == 0:
            self.attn_args['attn_type'] = 'full'

        # patch embedding
        layers = [
            PatchEmbed(patch_size, nx, ny, in_chans=in_dim, embed_dim=dim, ape=ape,
                       **self.patch_embed_args)
        ]
        for dpr in dprs:
            layers.append(AttnBlock(
                dim, num_heads, drop_path=dpr, seq_len=seq_len, rpe=not ape,
                wx=nx, wy=ny,
                **self.attn_args
            ))
            layers.append(MlpBlock(dim, drop_path=dpr, **self.mlp_args))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = {'pos_embed', 'cls_token',
                    'norm.weight', 'norm.bias',
                    'norm_embed', 'head.bias',
                    'relative_position'}
        return no_decay

    def get_classifier(self):
        return self.head

    def forward(self, x):
        B = x.shape[0]
        outputs = []
        x, nx, ny = self.layer1((x, None, None))
        if "layer1" in self._out_features:
            outputs.append(
                x[:, self.Nglos[0]:].transpose(-2, -1).reshape(B, -1, nx, ny)
            )

        x = x[:, self.Nglos[0]:].transpose(-2, -1).reshape(B, -1, nx, ny)
        x, nx, ny = self.layer2((x, nx, ny))
        if "layer2" in self._out_features:
            outputs.append(
                x[:, self.Nglos[1]:].transpose(-2, -1).reshape(B, -1, nx, ny)
            )

        x = x[:, self.Nglos[1]:].transpose(-2, -1).reshape(B, -1, nx, ny)
        x, nx, ny = self.layer3((x, nx, ny))
        if "layer3" in self._out_features:
            outputs.append(
                x[:, self.Nglos[2]:].transpose(-2, -1).reshape(B, -1, nx, ny)
            )

        if self.layer4 is not None:
            x = x[:, self.Nglos[2]:].transpose(-2, -1).reshape(B, -1, nx, ny)
            x, nx, ny = self.layer4((x, nx, ny))
            if "layer4" in self._out_features:
                outputs.append(
                    x[:, self.Nglos[3]:].transpose(-2, -1).reshape(B, -1, nx, ny)
                )

        return outputs


def build_msvit_backbone(cfg):
    args = dict(
        img_size=cfg.INPUT.MAX_SIZE_TRAIN,
        drop_rate=cfg.MODEL.TRANSFORMER.DROP,
        drop_path_rate=cfg.MODEL.TRANSFORMER.DROP_PATH,
        norm_embed=cfg.MODEL.TRANSFORMER.NORM_EMBED,
        avg_pool=cfg.MODEL.TRANSFORMER.AVG_POOL,
        freeze_at=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        out_features=cfg.MODEL.TRANSFORMER.OUT_FEATURES
    )
    args['arch'] = cfg.MODEL.TRANSFORMER.MSVIT.ARCH
    args['sharew'] = cfg.MODEL.TRANSFORMER.MSVIT.SHARE_W
    args['attn_type'] = cfg.MODEL.TRANSFORMER.MSVIT.ATTN_TYPE
    args['share_kv'] = cfg.MODEL.TRANSFORMER.MSVIT.SHARE_KV
    args['only_glo'] = cfg.MODEL.TRANSFORMER.MSVIT.ONLY_GLOBAL
    args['sw_exact'] = cfg.MODEL.TRANSFORMER.MSVIT.SW_EXACT
    args['ln_eps'] = cfg.MODEL.TRANSFORMER.MSVIT.LN_EPS
    args['mode'] = cfg.MODEL.TRANSFORMER.MSVIT.MODE

    return MsViT(**args)


class ViTHead(nn.Module):
    def __init__(
        self,
        in_dim, layer_cfgstr, input_size=14,
        qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_embed=False, **args
    ):
        super(ViTHead, self).__init__()
        if 'ln_eps' in args:
            ln_eps = args['ln_eps']
            self.norm_layer = partial(nn.LayerNorm, eps=ln_eps)
            logging.info("Customized LayerNorm EPS: {}".format(ln_eps))
        else:
            self.norm_layer = norm_layer
        self.drop_path_rate = drop_path_rate

        self.attn_args = dict({
            'attn_type': 'full', # full attention for head
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'drop': drop_rate,
            'attn_drop': attn_drop_rate,
            'norm_layer': norm_layer,
            'drop_path': drop_path_rate,
        })
        self.patch_embed_args = dict({
            'norm_layer': norm_layer,
            'norm_embed': norm_embed,
            'drop_rate': drop_rate,
        })
        self.mlp_args = dict({
            'mlp_ratio': 4.0,
            'norm_layer': norm_layer,
            'act_layer': nn.GELU,
            'drop': drop_rate,
            'drop_path': drop_path_rate,
        })

        layer_cfg = parse_arch(layer_cfgstr)
        layer_id, num_heads, dim, num_block, is_sparse_attn, nglo, patch_size, num_feats, ape \
            = layer_cfg['l'], layer_cfg['h'], layer_cfg['d'], layer_cfg['n'], \
              layer_cfg['s'], layer_cfg['g'], layer_cfg['p'], layer_cfg['f'], \
              layer_cfg['a']
        self.input_size = input_size
        self.nglo = nglo
        assert input_size%patch_size == 0, "Input size is not divided by patch size in ViTHead!"
        assert nglo == 0, "Number of global tokens in ViTHead is not 0!"
        nx = self.input_size // patch_size
        ny = self.input_size // patch_size
        seq_len = nx * ny + nglo

        # patch embedding
        layers = [
            PatchEmbed(patch_size, nx, ny, in_chans=in_dim, embed_dim=dim,
                       ape=ape, nglo=nglo, **self.patch_embed_args)
        ]
        for block_id in range(num_block):
            layers.append(AttnBlock(
                dim, num_heads, seq_len=seq_len, rpe=not ape,
                wx=nx, wy=ny, nglo=nglo,
                **self.attn_args
            ))
            layers.append(MlpBlock(dim,  **self.mlp_args))
        self.layer4 = nn.Sequential(*layers)
        self.norm = norm_layer(dim)
        self.out_channels = dim

    def forward(self, x):
        B, C, nx, ny = x.shape
        assert nx == ny == self.input_size, "Input size does not match the initialized size in ViThead!"
        nglo = self.nglo
        x, nx, ny = self.layer4((x, None, None))
        x = self.norm(x)
        x = x[:, nglo:].transpose(-2, -1).reshape(B, -1, nx, ny)
        return x
