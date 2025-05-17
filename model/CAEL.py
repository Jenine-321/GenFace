import torch
from torch import nn, einsum
import torch.nn.functional as F
import cv2
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .efficient_net.efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
from scipy.fftpack import dct, idct

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d

# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# attention

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context),
                                dim=1)  # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# transformer encoder, for small and large patches

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


# MultiGrainedCrossAttention
class MultiGrainedCrossAttention(nn.Module):

    def __init__(self, fg_dim, cg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(fg_dim, cg_dim,
                             PreNorm(cg_dim, Attention(cg_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ProjectInOut(cg_dim, fg_dim,
                             PreNorm(fg_dim, Attention(fg_dim, heads=heads, dim_head=dim_head, dropout=dropout)))
            ]))

    def forward(self, fg_tokens, cg_tokens):
        (fg_cls, fg_patch_tokens), (cg_cls, cg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]),
                                                                   (fg_tokens, cg_tokens))

        for fg_attend_lg,cg_attend_fg in self.layers:
            fg_cls = fg_attend_lg(fg_cls, context=cg_patch_tokens, kv_include_self=True) + fg_cls
            cg_cls = cg_attend_fg (cg_cls, context=fg_patch_tokens, kv_include_self=True) + cg_cls

        fg_tokens = torch.cat((fg_cls, fg_patch_tokens), dim=1)
        cg_tokens = torch.cat((cg_cls, cg_patch_tokens), dim=1)
        
        return fg_tokens, cg_tokens

class AppearanceEdgeCrossAttention(nn.Module):

    def __init__(self, fg_dim, cg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(cg_dim, fg_dim,
                             PreNorm(fg_dim, Attention(fg_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ProjectInOut(cg_dim, cg_dim,
                             PreNorm(cg_dim, Attention(cg_dim, heads=heads, dim_head=dim_head, dropout=dropout)))
            ]))

    def forward(self, fg_tokens, cg_tokens, edge_tokens):
        (fg_cls, fg_patch_tokens), (cg_cls, cg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (fg_tokens, cg_tokens))
        edge_cls = edge_tokens[:, 0:1,:]
        edge_patch_tokens = edge_tokens[:,1:,:]

        for edge_attend_fg, edge_attend_cg in self.layers:
            edge_attend_fg_cls= edge_attend_fg(edge_cls, context=fg_patch_tokens, kv_include_self=True) + edge_cls
            edge_attend_cg_cls = edge_attend_cg(edge_cls, context=cg_patch_tokens, kv_include_self=True) + edge_cls

        edge_attend_cls =  edge_attend_fg_cls + edge_attend_cg_cls
        edge_tokens = torch.cat((edge_attend_cls , edge_patch_tokens), dim=1)

        return edge_tokens

# MultiGrainedAppeaEdgeTransformer

class MultiGrainedAppeaEdgeTransformer(nn.Module):
    def __init__(
            self,
            *,
            depth,
            fg_dim,
            cg_dim,
            fg_enc_params,
            cg_enc_params,
            cross_attn_heads,
            cross_attn_depth,
            cross_ae_attn_depth,
            cross_attn_dim_head=64,
            dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            self.layers.append(nn.ModuleList([
                Transformer(dim=fg_dim, dropout=dropout, **fg_enc_params),  # FineGrainedTransformer
                Transformer(dim=cg_dim, dropout=dropout, **cg_enc_params),  # CoarseGrainedTransformer
                Transformer(dim=cg_dim, dropout=dropout, **cg_enc_params),  # EdgeTransformer
                MultiGrainedCrossAttention(fg_dim=fg_dim, cg_dim=cg_dim, depth=cross_attn_depth, heads=cross_attn_heads,
                                 dim_head=cross_attn_dim_head, dropout=dropout),
                AppearanceEdgeCrossAttention(fg_dim=fg_dim, cg_dim=cg_dim, depth= cross_ae_attn_depth, heads=cross_attn_heads,
                                 dim_head=cross_attn_dim_head, dropout=dropout),
            ]))

    def forward(self, fg_tokens, cg_tokens, edge_tokens):

        for fg_enc, cg_enc,edge_enc, mg_cross_atten, ae_cross_atten in self.layers:

            fg_tokens, cg_tokens,edge_tokens = fg_enc(fg_tokens), cg_enc(cg_tokens),edge_enc(edge_tokens)
            fg_tokens, cg_tokens = mg_cross_atten(fg_tokens, cg_tokens)
            edge_tokens = ae_cross_atten(fg_tokens, cg_tokens,edge_tokens)

        return fg_tokens, cg_tokens, edge_tokens


# patch-based image to token embedder

class ImageEmbedder(nn.Module):
    def __init__(
            self,
            *,
            dim,
            image_size,
            patch_size,
            dropout=0.,
            efficient_block=8,
            channels
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficient_net.delete_blocks(efficient_block)
        self.efficient_block = efficient_block

        for index, (name, param) in enumerate(self.efficient_net.named_parameters()):
            param.requires_grad = True

        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.efficient_net.extract_features_at_block(img, self.efficient_block)
        '''
        x_scaled = []
        for idx, im in enumerate(x):
            im = im.cpu().detach().numpy()
            for patch_idx, patch in enumerate(im):
                patch = 2.*(patch - np.min(patch))/np.ptp(patch)-1
                im[patch_idx] = patch

            x_scaled.append(im)
        x = torch.tensor(x_scaled).cuda()    
        '''
        # x = torch.tensor(x).cuda()
        '''
        for idx, im in enumerate(x):
            im = im.cpu().detach().numpy()
            for patch_idx, patch in enumerate(im):
                cv2.imwrite("patches/patches_"+str(idx)+"_"+str(patch_idx)+".png", patch)
        '''
        x = self.to_patch_embedding(x)
        # print("patch", x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

class EdgeEmbedder(nn.Module):
    def __init__(
            self,
            *,
            dim,
            image_size,
            patch_size,
            dropout=0.,
            efficient_block=8,
            channels
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.bn = nn.BatchNorm2d(channels),

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, img):
        x_edge_fea = self.features (img)
        x = self.to_patch_embedding(x_edge_fea)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)
# CAEL

class CAEL(nn.Module):
    def __init__(
            self,
            *,
            config
    ):
        super().__init__()
        image_size = config['model']['image-size']
        num_classes = config['model']['num-classes']
        fg_dim = config['model']['fg-dim']
        fg_channels = config['model']['fg-channels']
        cg_dim = config['model']['cg-dim']
        cg_channels = config['model']['cg-channels']
        fg_patch_size = config['model']['fg-patch-size']
        fg_enc_depth = config['model']['fg-enc-depth']
        fg_enc_heads = config['model']['fg-enc-heads']
        fg_enc_mlp_dim = config['model']['fg-enc-mlp-dim']
        fg_enc_dim_head = config['model']['fg-enc-dim-head']
        cg_patch_size = config['model']['cg-patch-size']
        cg_enc_depth = config['model']['cg-enc-depth']
        cg_enc_mlp_dim = config['model']['cg-enc-mlp-dim']
        cg_enc_heads = config['model']['cg-enc-heads']
        cg_enc_dim_head = config['model']['cg-enc-dim-head']
        cross_attn_depth = config['model']['cross-attn-depth']
        cross_ae_attn_depth =  config['model']['cross-ae-attn-depth']
        cross_attn_heads = config['model']['cross-attn-heads']
        cross_attn_dim_head = config['model']['cross-attn-dim-head']
        depth = config['model']['depth']
        dropout = config['model']['dropout']
        emb_dropout = config['model']['emb-dropout']

        self.fg_image_embedder = ImageEmbedder(dim=fg_dim, image_size=image_size, patch_size=fg_patch_size,
                                               dropout=emb_dropout, efficient_block=16, channels=fg_channels)
        self.cg_image_embedder = ImageEmbedder(dim=cg_dim, image_size=image_size, patch_size=cg_patch_size,
                                               dropout=emb_dropout, efficient_block=1, channels=cg_channels)
        self.edge_embedder = EdgeEmbedder(dim=cg_dim, image_size=image_size, patch_size=cg_patch_size,
                                               dropout=emb_dropout, efficient_block=1, channels=cg_channels)

        self.multi_grained_appea_edge_transformer = MultiGrainedAppeaEdgeTransformer(
            depth=depth,
            fg_dim=fg_dim,
            cg_dim=cg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            cross_ae_attn_depth =cross_ae_attn_depth,
            fg_enc_params=dict(
                depth=fg_enc_depth,
                heads=fg_enc_heads,
                mlp_dim=fg_enc_mlp_dim,
                dim_head=fg_enc_dim_head
            ),
            cg_enc_params=dict(
                depth=cg_enc_depth,
                heads=cg_enc_heads,
                mlp_dim=cg_enc_mlp_dim,
                dim_head=cg_enc_dim_head
            ),
            dropout=dropout
        )

        self.fg_mlp_head = nn.Sequential(nn.LayerNorm(fg_dim), nn.Linear(fg_dim, num_classes))
        self.cg_mlp_head = nn.Sequential(nn.LayerNorm(cg_dim), nn.Linear(cg_dim, num_classes))
        self.edge_mlp_head = nn.Sequential(nn.LayerNorm(cg_dim), nn.Linear(cg_dim, num_classes))

    def forward(self, img):
        numpy_array = img.detach().cpu().numpy()
        sobel_images = []
        rgb_images = []
        # 对数组进行迭代，逐个取出并转换为灰度图像
        for i in range(numpy_array.shape[0]):
            image = numpy_array[i]
            rgb_image = image / 255
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
            # 计算梯度的幅值，即特征图
            image_edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            image_edge = image_edge / 255
            sobel_images.append(image_edge)
            rgb_images.append(rgb_image)

        img_edge_t = torch.tensor(sobel_images).unsqueeze(1).float().cuda()  # 将灰度图像列表转换为张量，并增加维度为 (b, 1, 224, 224)
        img_rgb_t = torch.tensor(rgb_images).cuda()
        img_rgb_t  = img_rgb_t.permute(0, 3, 2, 1).float()

        fg_tokens = self.fg_image_embedder(img_rgb_t)  #
        cg_tokens = self.cg_image_embedder(img_rgb_t)  #
        edge_tokens = self.edge_embedder(img_edge_t)

        fg_tokens, cg_tokens, edge_tokens = self.multi_grained_appea_edge_transformer(fg_tokens, cg_tokens, edge_tokens)

        fg_cls, cg_cls, edge_cls = map(lambda t: t[:, 0], (fg_tokens, cg_tokens, edge_tokens))

        fg_logits = self.fg_mlp_head(fg_cls)
        cg_logits = self.cg_mlp_head(cg_cls)
        edge_logits = self.edge_mlp_head(edge_cls)

        return fg_logits + cg_logits + edge_logits
#
if __name__ == '__main__':

     import yaml

     config_path = "./configs/CAEL.yaml"
     with open(config_path, 'r') as ymlfile:
         config = yaml.safe_load(ymlfile)

     model = CAEL(config=config).cuda()
     x = torch.randn(2, 224, 224,3).cuda()
     y = model(x)
     checkpoint_path = '/home/zyn/disk1/FaceManipulationDetection/deepfake/CLIP_baseline/checkpoint/diffacetrain_CAEL/best.pth'
     state = torch.load(checkpoint_path)
     checkpoint = state['state_dict']  # Checkpoint = state['state_dict']
     model.load_state_dict(checkpoint)
     print("finished")
     print(y.shape)
