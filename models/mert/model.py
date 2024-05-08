from functools import partial
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
import timm.models.vision_transformer as timm_vit
from torchaudio.models import wav2vec2_model

# Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# MAE
class MAE(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        # MAE feature extractor
        self.feature_extractor, embed_dim = self._init_feature_extractor()

        if self.config.mae_frozen:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
        
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(config.mae_proj_dropout),
            nn.Linear(embed_dim, config.mae_tr_dim))

                
        # Temporal transformer
        self.temporal_token = nn.Parameter(torch.randn(1, 1, config.mae_tr_dim))

        dim_head = config.mae_tr_dim // config.mae_tr_nheads
        assert dim_head * config.mae_tr_nheads == config.mae_tr_dim, \
            "embed_dim must be divisible by num_heads"
        self.temporal_transformer = Transformer(
            dim = config.mae_tr_dim,
            depth = config.mae_tr_num_layers,
            heads = config.mae_tr_nheads,
            dim_head = dim_head,
            mlp_dim = config.mae_tr_dim_feedforward,
            dropout = config.mae_tr_dropout)
        
        # self.head_drop = nn.Dropout(config.mae_head_dropout)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(config.mae_tr_dim),
        #     nn.Linear(config.mae_tr_dim, config.mae_out_features)
        # )
        self.head = nn.Linear(config.mae_tr_dim, config.mae_out_features)


    def _init_feature_extractor(self):
        feature_extractor, embed_dim = vit_base_patch16(
            global_pool=self.config.mae_global_pool)
        feature_extractor = load_from_pretrain(
            model=feature_extractor,
            path=self.config.mae_pretrained_path,
            device=self.device,
            global_pool=self.config.mae_global_pool)
        return feature_extractor, embed_dim

    def forward(self, x):
        # x: B, F, C, H, W
        b = x.size(0)
        x  = rearrange(x, 'b f c h w -> (b f) c h w') # (BxF, C, H, W)

        x = self.feature_extractor(x)

        x = nn.functional.normalize(x, dim=-1)
        
        x = rearrange(x, '(b f) m -> b f m', b=b)

        x = self.proj(x)  # B, F, TR_DIM

        cls_temporal_tokens = repeat(
            self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        # if self.config.tr_global_pool == 'avg':
        #     x = x.mean(dim = 1)
        # else: 
        #     x = x[:, 0]

        # x = self.head_drop(x)
        # x = self.mlp_head(x) # B, F, OUT
        x = self.head(x[:,1:])

        return x


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        print('Interpolate positional embeding')
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int(
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" %
                  (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1,
                                            orig_size,
                                            orig_size,
                                            embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, 
                size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def get_state_dict(path, device):
    weights = torch.load(path, map_location=device)
    state_dict = weights['state_dict']
    out_state_dict = {
        key.removeprefix('model.'): value for key, value in state_dict.items()}
    return out_state_dict


def load_from_pretrain(model, path, device='cpu', global_pool=False):
    checkpoint_model = get_state_dict(path, device)
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    # if global_pool:
    #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    # else:
    #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    print(msg.missing_keys)
    return model


def vit_base_patch16(**kwargs):
    model = timm_vit.VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm,
            eps=1e-6),
        num_classes=0,  # self.head = nn.Identity()
        **kwargs)
    return model, 768


# Hubert

class HubertHead(nn.Module):
    """Head for Hubert classification task."""

    def __init__(self, hidden_size, head_dropout, out_features):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(head_dropout)
        self.out_proj = nn.Linear(hidden_size, out_features)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class HubertEncoder(nn.Module):
    def __init__(self,
            pretrained_model_path,
            out_features,
            head_dropout=0.,
            freeze_feature_extractor=True,
            device='cpu'):
        super().__init__()
        self.pretrained_model_path = pretrained_model_path
        self.device = device

        # Hubert
        self.hubert = self._init_hubert()
        # New
        self.head = nn.Linear(1024, out_features)
        
        if freeze_feature_extractor:
            for p in self.hubert.feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x : (batch, frames)
        x, _ = self.hubert(x)
        x = self.head(x)
        return x
    
    def _build_config(self):
        hubert_config = {
            'extractor_mode': 'layer_norm',
            'extractor_conv_layer_config': [
                (512, 10, 5),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2)],
            'extractor_conv_bias': True,
            'encoder_embed_dim': 1024,
            'encoder_projection_dropout': 0.0,
            'encoder_pos_conv_kernel': 128,
            'encoder_pos_conv_groups': 16,
            'encoder_num_layers': 24,
            'encoder_num_heads': 16,
            'encoder_attention_dropout': 0.1,
            'encoder_ff_interm_features': 4096,
            'encoder_ff_interm_dropout': 0.1,
            'encoder_dropout': 0.1,
            'encoder_layer_norm_first': True,
            'encoder_layer_drop': 0.1}
        return hubert_config

    def _init_hubert(self):
        config = self._build_config()
        hubert = wav2vec2_model(**config, aux_num_out=None)
        if self.pretrained_model_path is not None:
            state_dict = torch.load(
                self.pretrained_model_path,
                map_location=self.device)
            hubert.load_state_dict(state_dict)
            # TODO: Use logger
            print(f'Model loaded from {self.pretrained_model_path}')
        return hubert


class MERT(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config

        # Create Audio Encoder (Hubert)
        self.audio_encoder = HubertEncoder(
            pretrained_model_path=config.hubert_pretrained_model_path,
            out_features=config.hubert_out_features,
            head_dropout=config.hubert_head_dropout,
            freeze_feature_extractor=config.hubert_freeze_feature_extractor,
            device=device)
        self.audio_proj = nn.Sequential(
            nn.Linear(config.hubert_frames, config.num_frames),
            nn.LayerNorm(config.num_frames),
            nn.GELU())

        # Create Video Encoder (MAE-T)
        self.visual_encoder = MAE(config=self.config, device=device)
        self.visual_proj = nn.Sequential(
            nn.Linear(config.mae_frames, config.num_frames),
            nn.LayerNorm(config.num_frames),
            nn.GELU())

        # Cross transformer
        self.cross_token = nn.Parameter(torch.randn(1, 1, config.tr_dim))

        dim_head = config.tr_dim // config.tr_nheads
        assert dim_head * config.tr_nheads == config.tr_dim, \
            "embed_dim must be divisible by num_heads"
        self.cross_transformer = Transformer(
            dim = config.tr_dim,
            depth = config.tr_num_layers,
            heads = config.tr_nheads,
            dim_head = dim_head,
            mlp_dim = config.tr_dim_feedforward,
            dropout = config.tr_dropout)
        
        self.head_drop = nn.Dropout(config.head_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.tr_dim),
            nn.Linear(config.tr_dim, config.out_features)
        )

    def forward(self, audio, frames):
        b = audio.size(0)

        # Acoustic features
        audio_features = self.audio_encoder(audio)  # B, 251, H_OUT
        audio_features = nn.functional.normalize(audio_features, dim=-1)


        audio_features = self.audio_proj(audio_features.permute(0, 2, 1)).permute(0, 2, 1)  # B, F, H_OUT

        # Visual features
        visual_features = self.visual_encoder(frames) # B x 768, M_OUT
        visual_features = nn.functional.normalize(visual_features, dim=-1)

        visual_features = self.visual_proj(visual_features.permute(0, 2, 1)).permute(0, 2, 1) # B, F, M_OUT

        # Concatenation
        x = torch.concat([visual_features, audio_features], dim=-1)

        # Cross transformer
        cls_cross_tokens = repeat(
            self.cross_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_cross_tokens, x), dim=1)

        x = self.cross_transformer(x)

        if self.config.tr_global_pool == 'avg':
            x = x.mean(dim = 1)
        else: 
            x = x[:, 0]
        
        x = self.head_drop(x)
        x = self.mlp_head(x)

        return x

def main():
    from torchinfo import summary
    from engine import Config

    conf = Config.load_from_yaml(
        '/home/houdaifa.atou/main/code/emorec/configs/mert.yaml')
    model = MERT(conf.model, device='cpu')
    summary(model, input_size=[(8, 80640), (8, 16, 3, 224, 224)])

    import torchvision
    torchvision.models.vit_b_16


if __name__ == '__main__':
    main()
