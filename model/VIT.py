import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_


class EmbeddingLayer(nn.Module):
    def __init__(self, in_channels, embed_dim, img_size, patch_size):
        super().__init__()

        assert (img_size % patch_size == 0)

        self.num_tokens = (img_size // patch_size) ** 2 + 1
        self.embed_dim = embed_dim
        self.project = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))

        # Initialization of cls_token and pos_embed
        nn.init.normal_(self.cls_token, std=1e-6)
        trunc_normal_(self.pos_embed, std=2e-2)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[1]
                factor = (n+1) // 2
                if n % 2 == 1:
                    center = factor - 1
                else:
                    center = factor - 0.5
                og = np.ogrid[:n, :n]
                weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                m.weight.data.copy_(torch.from_numpy(weights_np))

    def forward(self, x):
        B, C, H, W = x.shape
        # Shape of embedding: (B, embed_dim, patch_size, patch_size)
        embedding = self.project(x)
        # Shape of z: (B, patch_size^2, embed_dim)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)

        # Shape of cls_token: (B, 1, embed_dim)
        cls_token = self.cls_token.expand(B, -1, -1)
        # Shape of z: (B, patch_size^2 + 1, embed_dim)
        z = torch.cat([cls_token, z], dim=1)
        # Through training, cls_token includes the information of object classes
        # for the final prediction.

        # Positional embedding
        z = z + self.pos_embed
        return z

class MSA(nn.Module):
    def __init__(self, dim=192, num_heads=12, qkv_bias=False, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads...'

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[1]
                factor = (n+1) // 2
                if n % 2 == 1:
                    center = factor - 1
                else:
                    center = factor - 0.5
                og = np.ogrid[:n, :n]
                weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                m.weight.data.copy_(torch.from_numpy(weights_np))

    def forward(self, x):
        B, N, C = x.shape
        # C is dim.
        qkv = self.qkv(x)
        # qkv shape: (B, N, 3 * dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv shape: (3, B, self.num_heads, N, dim // self.num_heads)

        q, k, v = qkv.unbind(0)
        # Shape of q, k, v: (B, self.num_heads, N, dim // self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Shape of q: (B, self.num_heads, N, dim // self.num_heads)
        # Shape of k.transpose(-2, -1): (B, self.num_heads, dim // self.num_heads, N)
        # Shape of q @ k.transpose(-2, -1): (B, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Shape of attn: (B, self.num_heads, N, N)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # Shape of (attn @ v): (B, self.num_heads, N, dim // self.num_heads)
        # Shape of (attn @ v).transpose(1, 2): (B, N, self.num_heads, dim // self.num_heads)
        # Shape of (attn @ v).transpose(1, 2).reshape(B, N, C): (B, N, dim)
        x = self.proj(x)
        # Shape of x: (B, N, dim)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[1]
                factor = (n+1) // 2
                if n % 2 == 1:
                    center = factor - 1
                else:
                    center = factor - 0.5
                og = np.ogrid[:n, :n]
                weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                m.weight.data.copy_(torch.from_numpy(weights_np))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                        attn_drop=attn_drop, proj_drop=drop)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[1]
                factor = (n+1) // 2
                if n % 2 == 1:
                    center = factor - 1
                else:
                    center = factor - 0.5
                og = np.ogrid[:n, :n]
                weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                m.weight.data.copy_(torch.from_numpy(weights_np))

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x

class ViT(nn.Module):
    def __init__(self,
                 img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=12,
                 num_heads=12, mlp_ratio=2., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.patch_embed = EmbeddingLayer(in_channels, embed_dim=embed_dim,
                                          img_size=img_size,
                                          patch_size=patch_size)
        self.blocks = nn.Sequential(
            *[EncoderBlock(dim=embed_dim, num_heads=num_heads,
                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           act_layer=act_layer, norm_layer=norm_layer) for i in range(depth)]
        )

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[1]
                factor = (n+1) // 2
                if n % 2 == 1:
                    center = factor - 1
                else:
                    center = factor - 0.5
                og = np.ogrid[:n, :n]
                weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                m.weight.data.copy_(torch.from_numpy(weights_np))

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        # Shape of x: [bs, N, embed_dim]
        x = self.norm(x)
        x = self.head(x)[:, 0]
        # Shape of output of self.head(x): [bs, N, num_classes]
        # Shape of self.head(x)[:, 0]: [bs, num_classes]
        # Why do we select the first row of self.head(x) like self.head(x)[:, 0]?
        # -> In the EmbeddingLayer,
        #    we add cls_token to the patch embedding vectors at the first row
        #    like z = torch.cat([cls_token, z], dim=1).
        #    Therefore, the first row of self.head(x) includes the information of object classes.
        return x


if __name__=='__main__':
    img_test = torch.randn([2, 3, 32, 32])
    model= ViT()
    out = model(img_test)
