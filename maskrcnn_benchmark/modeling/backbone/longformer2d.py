# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
# Written by Pengchuan Zhang, penzhan@microsoft.com
import random
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from .slidingchunk_2d import slidingchunk_2d, mask_invalid_locations, slidingchunk_2dautograd


class Long2DSCSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., w=7, d=1,
                 autoregressive=False, sharew=False, nglo=1, only_glo=False, exact=0, autograd=False, rpe=False,
                 mode=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.Nglo = nglo
        self.only_glo = only_glo
        if self.only_glo:
            assert self.Nglo >= 1, "Nglo == 0 in the only global mode!"

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if nglo >= 1:
            if sharew:
                self.query_global = self.query
                self.kv_global = self.kv
                self.proj_global = self.proj
            else:
                self.query_global = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
                self.proj_global = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attention_window = w
        self.attention_dilation = d
        self.autoregressive = autoregressive

        assert self.attention_dilation == 1, "Dilation is not supported!"
        assert not self.autoregressive, "Autoregressive is not supported yet!"
        self.exact = exact
        # use autograd or handgrad
        self.longform2d_mm = slidingchunk_2dautograd if autograd else slidingchunk_2d

        # Inspired by swin transformer:
        # https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L88-L103
        # define parameter tables for local and global relative position bias
        self.rpe = rpe
        if rpe:
            self.local_relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * 2 * w - 1) * (2 * 2 * w - 1), num_heads))  # (4*w-1, 4*w-1, nH)
            trunc_normal_(self.local_relative_position_bias_table, std=.02)
            if nglo >= 1:
                self.g2l_relative_position_bias = nn.Parameter(
                    torch.zeros(2, num_heads, nglo))  # (2, nH, nglo)
                self.g2g_relative_position_bias = nn.Parameter(
                    torch.zeros(num_heads, nglo, nglo))  # (nH, nglo, nglo)
                trunc_normal_(self.g2l_relative_position_bias, std=.02)
                trunc_normal_(self.g2g_relative_position_bias, std=.02)

            # get pair-wise relative position index
            coords_h = torch.arange(-w, 2*w)
            coords_w = torch.arange(-w, 2*w)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, 3w, 3w
            coords_unfold = rearrange(
                coords, 'c (m x) (n y) -> c m n (x y)', x=w, y=w
            )  # 2, 3, 3, 9w^2
            q_coords = coords_unfold[:, 1, 1, :] # 2, w^2
            relative_coords = torch.cat([
                # -1, -1
                q_coords[:, :, None] - coords_unfold[:, 0, 0, :][:, None, :],
                # -1, 0
                q_coords[:, :, None] - coords_unfold[:, 0, 1, :][:, None, :],
                # -1, 1
                q_coords[:, :, None] - coords_unfold[:, 0, 2, :][:, None, :],
                # 0,-1
                q_coords[:, :, None] - coords_unfold[:, 1, 0, :][:, None, :],
                # 0,0
                q_coords[:, :, None] - q_coords[:, None, :],
                # 0,1
                q_coords[:, :, None] - coords_unfold[:, 1, 2, :][:, None, :],
                # 1, -1
                q_coords[:, :, None] - coords_unfold[:, 2, 0, :][:, None, :],
                # 1, 0
                q_coords[:, :, None] - coords_unfold[:, 2, 1, :][:, None, :],
                # 1, 1
                q_coords[:, :, None] - coords_unfold[:, 2, 2, :][:, None, :],
            ], dim=-1)  # 2, w^2, 9w^2
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # w^2, 9w^2, 2
            relative_coords[:, :, 0] += 2 * w - 1  # shift to start from 0
            relative_coords[:, :, 1] += 2 * w - 1
            relative_coords[:, :, 0] *= 2 * 2 * w - 1
            relative_position_index = relative_coords.sum(-1)  # w^2, 9w^2
            self.register_buffer("relative_position_index", relative_position_index)

        # mode to control the sampling strategy of neighbor blocks
        # 0: all 8 blocks; -1: no neighbor block; >0: random sample one block
        self.mode = mode

    def forward(self, x, nx, ny):
        B, N, C = x.shape
        Nloc = nx * ny
        Nglo, H, M, W = self.Nglo, self.num_heads, self.head_dim, self.attention_window
        W2 = W ** 2
        assert Nglo + Nloc == N, "Global dimension does not match!"

        # get the mode of the longformer attention
        mode = self.mode
        kv_nums = 9 * W2
        if self.mode > 0:
            if self.training:
                mode = random.randrange(1, 9)  # 1 <= mode <= 8
                kv_nums = 2 * W2
            else:
                mode = 0  # full during evaluation
        elif mode == -1:
            kv_nums = W2

        # compute the local attention
        q = self.scale * self.query(x[:, Nglo:]).reshape(B, Nloc, H, M).transpose(1, 2).contiguous()
        kv = self.kv(x).reshape(B, N, 2, H, M).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        if self.only_glo:
            # local to global attn10: (B, self.num_heads, Nloc, Nglo)
            attn1 = torch.bmm(q.view(B*H, Nloc, M), k[:, :, :Nglo].reshape(B*H, Nglo, M).transpose(-2, -1)).view(B, H, Nloc, Nglo)
        else:
            (q_img, k_img, v_img) = map(
                lambda t: rearrange(t, 'b h (x y) c -> (b h) c x y', x=nx),
                (q, k[:, :, Nglo:], v[:, :, Nglo:]))
            # pad 0's to make sure that nx % W == 0, ny % W == 0
            (padx, pady) = map(lambda t: (W - t % W) % W, (nx, ny))
            (mx, my) = map(lambda t: (t[0] + t[1]) // W,
                           ((nx, padx), (ny, pady)))
            if padx > 0 or pady > 0:
                (q_img, k_img, v_img) = map(
                    lambda t: F.pad(t, (0, pady, 0, padx)), (q_img, k_img, v_img)
                )
            # unfold the padded tensor
            (q_img, k_img, v_img) = map(
                lambda t: rearrange(t, 'b c (m x) (n y) -> b c m n (x y)', x=W, y=W),
                (q_img, k_img, v_img)
            )

            # local to global attn10: (B*H, mx, my, w^2, Nglo)
            attn10 = einsum('b c m n l, b t c -> b m n l t', q_img,
                       k[:, :, :Nglo].reshape(B*H, Nglo, M))
            # local to local attn11： (B*H, mx, my, W**2, 9*W**2), mode = 0
            # attn11： (B*H, mx, my, W**2, W**2), mode = -1
            # attn11： (B*H, mx, my, W**2, 2*W**2), mode > 0
            attn11 = self.longform2d_mm(q_img, k_img, False, mode)

            if self.rpe:
                if Nglo >= 1:
                    # local to global bias
                    attn10 = attn10 + self.g2l_relative_position_bias[1].unsqueeze(0).expand(B, -1, -1).reshape(B*H, Nglo)[:, None, None, None, :]
                # local to local bias
                if mode == -1:
                    relative_position_index = self.relative_position_index[:, 4 * W2:5 * W2].contiguous()
                elif mode == 0:
                    relative_position_index = self.relative_position_index
                else:  # mode > 0
                    chunk_id = mode if mode > 4 else mode - 1
                    relative_position_index = torch.cat([
                        self.relative_position_index[:, 4 * W2:5 * W2],
                        self.relative_position_index[:, chunk_id * W2:(chunk_id+1) * W2],
                    ], dim=-1)
                local_relative_position_bias = self.local_relative_position_bias_table[
                    relative_position_index.view(-1)].view(1, W2, kv_nums, -1)  # w^2, kv_nums,H
                local_relative_position_bias = local_relative_position_bias.permute(
                    0, 3, 1, 2).expand(B, -1, -1, -1).contiguous().view(B*H, W2, kv_nums)  # B*H, w^2, kv_nums
                attn11 = attn11 + local_relative_position_bias[:, None, None, :, :]

            num_invalid = mask_invalid_locations(
                attn11, mx, my, padx, pady, W, exact=self.exact, mode=mode
            )
            attn1 = torch.cat((attn10, attn11), dim=-1)

        attn1 = (attn1 - torch.max(attn1, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        # update x1: (B, self.num_heads, Nloc, self.head_dim)
        if self.only_glo:
            x1 = torch.bmm(
                attn1.view(B * H, Nloc, Nglo), v[:, :, :Nglo].reshape(B * H, Nglo, M)
            ).view(B, H, Nloc, M)
        else:
            attnl2g = attn1[:, :, :, :, :Nglo]
            x1 = self.longform2d_mm(attn1[:, :, :, :, Nglo:Nglo+kv_nums], v_img, True, mode)
            if Nglo >= 1:
                x1 = x1 + einsum(
                    'b m n l t, b t c -> b c m n l', attnl2g,
                    v[:, :, :Nglo].reshape(B * H, Nglo, M)
                )
            x1 = rearrange(x1, 'b c m n (x y) -> b (m x) (n y) c', x=W)
            x1 = x1[:, :nx, :ny].reshape(B, H, Nloc, M)
        x1 = x1.transpose(1, 2).reshape(B, Nloc, C)

        try:
            x1 = self.proj(x1)
        except RuntimeError as e:
            # guard against possible half vs float error
            x1 = self.proj(x1.float())

        if Nglo == 0:
            return self.proj_drop(x1)

        # compute the glocal attention; same with vanilla multi-head attention
        q_global = self.scale * self.query_global(x[:, :Nglo]).reshape(B, Nglo, H, M).transpose(1, 2)
        kv_global = self.kv_global(x).reshape(B, N, 2, H, M).permute(2, 0, 3, 1, 4)
        k_global, v_global = kv_global[0], kv_global[1]  # make torchscript happy (cannot use tensor as tuple)
        # attention matrix
        attn0 = torch.bmm(q_global.reshape(B*H, Nglo, M), k_global.reshape(B*H, N, M).transpose(-2, -1))
        if self.rpe:
            # relative position embedding of global tokens
            global_relative_position_bias = torch.cat([
                self.g2g_relative_position_bias,
                self.g2l_relative_position_bias[0].unsqueeze(-1).expand(-1, -1, Nloc)
            ], dim=-1)  # nH, nglo, N
            attn0 = attn0 + global_relative_position_bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B*H, Nglo, N)

        attn0 = (attn0 - torch.max(attn0, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn0 = self.attn_drop(attn0)
        # context vector
        x0 = torch.bmm(attn0, v_global.reshape(B*H, N, M)).view(B, H, Nglo, M).transpose(1, 2).reshape(B, Nglo, C)
        x0 = self.proj_global(x0)

        return self.proj_drop(torch.cat((x0, x1), dim=1))

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        S = T
        Nglo, H, M, W = module.Nglo, module.num_heads, module.head_dim, module.attention_window
        macs = 0
        n_params = 0

        # Sliding window scaled-dot-product macs
        if module.only_glo:
            # local to global
            # [B x T x (C-Nglo)] x [B x C x Nglo] --> [B x T x Nglo]
            num_macs_kq = (C - Nglo) * Nglo * C
        else:
            # local to local
            # [B x T x (C-Nglo)] x [B x C x (S-Nglo)] --> [B x (C-Nglo) x (9 * W**2)]
            num_macs_kq = (C-Nglo) * (9 * W**2) * C
            # local to global
            # [B x T x (C-Nglo)] x [B x C x Nglo] --> [B x T x Nglo]
            num_macs_kq += (C-Nglo) * Nglo * C
        # global to all
        # [B x T x Nglo] x [B x C x S] --> [B x Nglo x S]
        num_macs_kq += Nglo * S * C
        # same computational cost for attn * v -> context
        num_macs_v = num_macs_kq

        macs += num_macs_kq + num_macs_v
        # print('macs att', macs / 1e8)

        # self attention: T should be equal to S
        assert T == S
        # by default, we share weights for local and global tokens
        q_params = sum([p.numel() for p in module.query.parameters()])
        kv_params = sum([p.numel() for p in module.kv.parameters()])
        n_params += q_params + kv_params
        # multiply by Seq length
        macs += (q_params + kv_params) * T
        # print('macs qkv', qkv_params * T / 1e8)

        # by default, we share weights for local and global tokens
        proj_params = sum([p.numel() for p in module.proj.parameters()])
        n_params += proj_params
        macs += (proj_params * T)
        # print('macs proj', proj_params * T / 1e8)

        module.__flops__ += macs
        # return n_params, macs
