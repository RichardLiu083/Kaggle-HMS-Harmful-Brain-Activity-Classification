import torch
import torch.nn as nn
from torchvision import models
import rotary_embedding_torch
from rotary_embedding_torch import apply_rotary_emb
import math
from einops import rearrange


class WidthAttention(nn.Module):
    def __init__(self, in_ch, width: int, debug_mode=False):
        super().__init__()
        h_dim = 64
        self.attention = nn.Sequential(  # B, w
            nn.Conv2d(in_ch, h_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(h_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(h_dim, width),
            nn.Sigmoid()
        )
        self.feat_atten = None
        if not debug_mode:
            self.attention.register_forward_hook(self._capture_attention)

    def _capture_attention(self, module, input, output):
        self.feat_atten = output

    def forward(self, x):
        attention = self.attention(x)
        attention = attention.unsqueeze(1).unsqueeze(1)
        return x * attention


class MultiHeadAttention(nn.Module):
    def __init__(self, in_ch, width: int, heads: int, debug_mode=False):
        super().__init__()
        self.attentions = nn.ModuleList([WidthAttention(in_ch // (heads * 2), width, debug_mode) for _ in range(heads)])
        assert in_ch % heads == 0, f'in_ch: {in_ch} must be divisible by heads: {heads}'
        self.projections = nn.ModuleList([nn.Conv2d(in_ch, in_ch // (heads * 2), kernel_size=(1, 1)) for _ in range(heads)])
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch // 4, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_ch // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch // 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(in_ch // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        heads = []
        for i, atten in enumerate(self.attentions):
            head = self.projections[i](x)
            head = atten(head)
            heads.append(head)
        return self.fuse(torch.cat(heads, 1))


class EfficientNet(nn.Module):
    def __init__(self, width: int, in_ch=4, num_classes=6, weights='IMAGENET1K_V1', use_attention=False, debug_mode=False, cnn_type='b0'):
        super(EfficientNet, self).__init__()
        if cnn_type == 'b0':
            efficientnet = models.efficientnet_b0(weights=weights)
            ori_net = list(efficientnet.features.children())
            cnn_ch = 1280
            c0_ch = 32
            w_factor = 5
        elif cnn_type == 'v2s':
            efficientnet = models.efficientnet_v2_s(weights=weights)
            ori_net = list(efficientnet.features.children())
            cnn_ch = 1280
            c0_ch = 24
            w_factor = 5
        elif cnn_type == 'convnext-tiny':
            network = models.convnext_tiny(weights=weights)
            ori_net = list(network.children())[0]
            cnn_ch = 768
            c0_ch = 96
            w_factor = 4
        elif cnn_type == 'mobilenet':
            network = models.mobilenet_v3_large(weights=weights)
            ori_net = list(network.children())[0]
            cnn_ch = 960
            c0_ch = 16
            w_factor = 5
        else:
            raise NotImplementedError(f'cnn_type: {cnn_type} not implemented')
        self.feat_atten = None
        if use_attention:
            w = width // (2 ** 5)
            self.width_attention = MultiHeadAttention(cnn_ch, w + 1, 4, debug_mode)
            ori_net.append(self.width_attention)
        self.features = nn.Sequential(*ori_net)
        # self.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        stride = 2 ** (5 - w_factor + 1)
        self.features[0][0] = nn.Conv2d(in_ch, c0_ch, kernel_size=(17, 17), stride=(stride, stride), padding=(8, 8), bias=False)
        # self.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=(11, 11), stride=(2, 2), padding=(6, 6), bias=False)
        self.adv_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(cnn_ch, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adv_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiHeadRotaryAttention(nn.Module):
    def __init__(self, h_dim: int, heads: int, is_2d=False):
        super().__init__()
        self.heads = heads
        self.h_dim = h_dim
        self.is_2d = is_2d
        self.rope = rotary_embedding_torch.RotaryEmbedding(dim=h_dim // heads // 2)
        if is_2d:
            self.projection = nn.Conv2d(h_dim, h_dim, kernel_size=(1, 1))
        else:
            self.projection = nn.Linear(h_dim, h_dim * 3)
        self.out = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
        )

    def forward(self, x):
        xp = self.projection(x)
        if self.is_2d:
            q, k, v = map(lambda t: rearrange(t, 'b (hs d) h w -> b hs h w d', hs=self.heads), [xp, xp, xp])
            q, k = map(self.rope.rotate_queries_or_keys, [q, k])
            q, k, v = map(lambda t: rearrange(t, 'b hs h w d -> b hs (h w) d', hs=self.heads), [q, k, v])
            attention = torch.einsum('b h i d, b h j d -> b h i j', q, k) * (self.h_dim ** -0.5)
            attention = torch.nn.functional.softmax(attention, dim=-1)
            xv = torch.einsum('b h i j, b h j c -> b h i c', attention, v)
            x = rearrange(xv, 'b h s c -> b s (h c)') + rearrange(x, 'b c h w -> b (h w) c')
        else:
            q, k, v = xp.chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b s (h d) -> b h s d', h=self.heads), [q, k, v])
            q, k = map(self.rope.rotate_queries_or_keys, [q, k])
            attention = torch.einsum('b h i d, b h j d -> b h i j', q, k) * (self.h_dim ** -0.5)
            attention = torch.nn.functional.softmax(attention, dim=-1)
            xv = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
            x = rearrange(xv, 'b h s d -> b s (h d)') + x
        return self.out(x)


class EfficientTransNet(nn.Module):
    def __init__(self,
                 in_ch=4,
                 num_classes=6,
                 hid_dim=128,
                 weights='IMAGENET1K_V1',
                 debug_mode=False,
                 width=656 + 256,
                 pe_type='rotary',
                 use_pe_2d=False,
                 heads=8
         ):
        super(EfficientTransNet, self).__init__()
        efficientnet = models.efficientnet_b0(weights=weights)
        ori_net = list(efficientnet.features.children())
        self.feat_atten = None
        self.features = nn.Sequential(*ori_net[:-2])
        self.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=(17, 17), stride=(2, 2), padding=(8, 8), bias=False)
        self.cnn_proj = nn.Sequential(
            nn.Conv2d(192, hid_dim, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(hid_dim),
            nn.GELU(),
        )
        # Transformer related
        # Create class token with 2d PE
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim), requires_grad=True)
        # self.cls_token = nn.Parameter(torch.zeros((1, hid_dim, 1, self.get_size(width))), requires_grad=True)
        self.heads = heads
        self.hid_dim = hid_dim
        self.pe_type = pe_type
        if pe_type == 'rotary':
            self.mha = MultiHeadRotaryAttention(hid_dim, self.heads, is_2d=use_pe_2d)
        else:
            self.mha = nn.MultiheadAttention(hid_dim, self.heads)
        self.trans_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hid_dim, nhead=self.heads, dim_feedforward=hid_dim, dropout=0.1, batch_first=True),
            num_layers=2
        )
        # output layer
        self.cls_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hid_dim, hid_dim // 4),
            nn.Linear(hid_dim // 4, num_classes, bias=False)
        )

    def generate_positional_encoding(self, seq_len):
        hid_dim = self.hid_dim
        # Initialize the positional encoding matrix
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2).float() * (-math.log(10000.0) / hid_dim))
        pe = torch.zeros(seq_len, hid_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def get_size(w, level=5):
        for _ in range(level):
            b = 1 if w % 2 == 1 else 0
            w = int(w / 2 + b)
        return int(w)

    def forward(self, x):
        # extract CNN features
        x = self.features(x)  # [B, 192, 4, w]
        x = self.cnn_proj(x)  # [B, 256, 4, w]
        # Transformer encoder related operations
        # Project and convert to channel-last
        x = rearrange(x, 'b c h w -> b (h w) c')
        # Append class token
        cls_token = self.cls_token.repeat_interleave(x.size(0), 0)
        x = torch.cat([cls_token, x], 1)
        if self.pe_type == 'rotary':
            x = self.mha(x)
        else:
            pe = self.generate_positional_encoding(x.size(1)).to(x.device)
            v, _ = self.mha[1](x + pe, x + pe, x + pe)
            x = v + x
        x = self.trans_encoder(x)[:, 0]
        # projection head
        x = self.cls_head(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    width = 300 + 256
    in_ch = 1
    model = EfficientNet(in_ch=in_ch, width=width, use_attention=True, debug_mode=True, cnn_type='mobilenet')
    model(torch.randn(2, in_ch, 400, width))
    s = summary(model, input_size=(1, in_ch, 400, width), device='cpu')

    # model = EfficientTransNet(in_ch, width=width, debug_mode=True)
    # model(torch.randn(2, in_ch, 400, width))
    # s = summary(model, input_size=(1, in_ch, 400, width), device='cpu')
    #
    # from torchvision.datasets import CIFAR10
    # import torchvision.transforms as transforms
    # from torch.utils.data import DataLoader
    # from tqdm import tqdm
    # import mlflow
    # from utils import seed_everything
    # from lion_pytorch import Lion
    # from prodigyopt import Prodigy
    # seed_everything(95277)
    # mlf_ip = 'localhost'
    # mlf_port = 5000
    # mlflow_url = f'http://{mlf_ip}:{mlf_port}'
    # mlflow.set_tracking_uri(mlflow_url)
    # experiment = mlflow.set_experiment('Model test')
    # if experiment is None:
    #     experiment = mlflow.create_experiment('Model test')
    #
    # # Define transformations
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((400, width), antialias=True),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    #
    # # Load the training and test datasets
    # bs = 16
    # dataset_train = CIFAR10(root='G:\dataset', train=True, download=True, transform=transform)
    # loader_train = DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=5, pin_memory=True)
    # pe = ['rotary', 'sincos'][0]
    # name = f'PE-{pe}-new-classqkv (P)'
    # with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f'{name}'):
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = EfficientTransNet(3, width=width, debug_mode=True, num_classes=10, pe_type=pe).to(device)
    #     # optim = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
    #     # optim = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    #     optim = Prodigy(model.parameters(), lr=1., weight_decay=1e-2)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(loader_train) * 10)
    #     scaler = torch.cuda.amp.GradScaler()
    #     for e in range(20):
    #         bar = tqdm(loader_train, total=len(loader_train), desc='Training')
    #         running_loss = 0
    #         for i, batch in enumerate(bar):
    #             img, label = batch
    #             with torch.cuda.amp.autocast(enabled=False):
    #                 out = model(img.to(device))
    #                 loss = nn.functional.cross_entropy(out, label.to(device))
    #             if not torch.isnan(loss).any():
    #                 running_loss = running_loss * 0.9 + loss.item() * 0.1 if running_loss != 0 else loss.item()
    #                 bar.set_postfix({'loss': running_loss})
    #             else:
    #                 print('Loss is nan')
    #                 with torch.cuda.amp.autocast(enabled=False):
    #                    model(img.to(device))
    #             scaler.scale(loss).backward()
    #             scaler.unscale_(optim)
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    #             scaler.step(optim)
    #             scaler.update()
    #             optim.zero_grad()
    #             if i > 150:
    #                 break
    #         scheduler.step()
    #         # Log if not nan
    #         if not math.isnan(running_loss):
    #             mlflow.log_metric('loss', running_loss, step=e)
    #         else:
    #             mlflow.log_metric('loss', -1e7, step=e)


