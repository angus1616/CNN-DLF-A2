
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from baseline import Base
from utils import *
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce



class VanillaCNN(Base):
    def __init__(self,
                 input_channels: int,
                 num_classes: int = 100,
                 model_name: str = 'Vanilla CNN',
                 model_run_no: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.best_model_path = None
        self.model_name = model_name
        self.model_run_no = model_run_no
        self.best_model_state_dict = None

        self.conv1 = nn.Conv2d(input_channels, 64, 5, 1, padding='same')
        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding='same')
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding='same')
        self.fc1 = nn.Linear(256*8*8, self.num_classes)

    # Forward pass
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = X.view(-1, 256*8*8)
        X = self.fc1(X)
        return F.log_softmax(X, dim=1)


class CNN(Base):
    def __init__(self,
                 input_channels: int,
                 activation: str = 'relu',
                 num_classes: int = 100,
                 model_name: str = 'Optimised CNN',
                 model_run_no: int = 1,
                 max_pool: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.model_run_no = model_run_no
        self.best_model_path = None
        self.best_model_state_dict = None

        self.conv1 = nn.Conv2d(input_channels, 64, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding='same')
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding='same')
        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding='same')
        self.conv5 = nn.Conv2d(512, 512, 3, 1, padding='same')
        self.fc1 = nn.Linear(512*2*2, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(.25)
        if max_pool == True:
          self.pool = nn.MaxPool2d(2,2)
        else:
          self.pool = nn.AvgPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        self.activation = act_func(activation)

    # Forward pass
    def forward(self, X):
        X = self.pool(self.activation(self.bn1(self.conv1(X))))
        X = self.pool(self.activation(self.bn2(self.conv2(X))))
        X = self.pool(self.activation(self.bn3(self.conv3(X))))
        X = self.pool(self.activation(self.bn4(self.conv4(X))))
        X = self.activation(self.bn5(self.conv5(X)))
        X = X.view(-1, 512*2*2)
        self.dropout(X)
        X = self.activation(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)



class ResidualBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = 'relu',
                 downsample: bool = False,
                 ) -> None:
        super().__init__()

        # For downsampling
        if downsample:
            self.conv1 = nn.Conv2d(
                in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_features)
            self.residual = nn.Sequential(
                nn.Conv2d(in_features, out_features,
                          kernel_size=1, stride=2),
                nn.BatchNorm2d(out_features)
            )
        # Non-downsampling
        else:
            self.conv1 = nn.Conv2d(
                in_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_features)
            self.residual = nn.Identity()

        self.conv2 = nn.Conv2d(out_features, out_features,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.activation = act_func(activation)

    # Forward pass
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = self.residual(input)
        input = self.activation(self.bn1(self.conv1(input)))
        input = self.activation(self.bn2(self.conv2(input)))
        input = input + residual
        return self.activation(input)

# Create class for ResNet


class ResNet12(Base):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 activation: str = 'relu',
                 model_name: str = "resnet12",
                 model_run_no: int = 1,
                 ) -> None:
        super().__init__()
        # init class variables
        self.model_name = model_name
        self.best_model_state_dict = None
        self.best_model_path = None
        self.model_run_no = model_run_no
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            act_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, downsample=False, activation=activation),
            ResidualBlock(64, 64, downsample=False, activation=activation),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, downsample=True, activation=activation),
            ResidualBlock(128, 128, downsample=False, activation=activation),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(128, 256, downsample=True, activation=activation),
            ResidualBlock(256, 256, downsample=False, activation=activation)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    # Forward pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet34(Base):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 activation: str = 'relu',
                 model_name: str = "resnet34",
                 model_run_no: int = 1,
                 ) -> None:
        super().__init__()
        # init class variables
        self.model_name = model_name
        self.best_model_state_dict = None
        self.best_model_path = None
        self.model_run_no = model_run_no
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            act_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, downsample=False, activation=activation),
            ResidualBlock(64, 64, downsample=False, activation=activation),
            ResidualBlock(64, 64, downsample=False, activation=activation)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, downsample=True, activation=activation),
            ResidualBlock(128, 128, downsample=False, activation=activation),
            ResidualBlock(128, 128, downsample=False, activation=activation),
            ResidualBlock(128, 128, downsample=False, activation=activation),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(128, 256, downsample=True, activation=activation),
            ResidualBlock(256, 256, downsample=False, activation=activation),
            ResidualBlock(256, 256, downsample=False, activation=activation),
            ResidualBlock(256, 256, downsample=False, activation=activation),
            ResidualBlock(256, 256, downsample=False, activation=activation),
            ResidualBlock(256, 256, downsample=False, activation=activation)
        )

        self.layer5 = nn.Sequential(
            ResidualBlock(256, 512, downsample=True, activation=activation),
            ResidualBlock(512, 512, downsample=False, activation=activation),
            ResidualBlock(512, 512, downsample=False, activation=activation)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    # Forward pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




#this transformer was adapted from https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
#the main difference being that I am comparing feature extraction techniques between the CONV and FCL

class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 32,
                 conv_feature_extractor: bool = True):

        self.patch_size = patch_size
        super().__init__()
        # use a convolutional layer to extract patches from the image
        if conv_feature_extractor:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, emb_size,
                          kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )
        else:
            self.projection = nn.Sequential(
                Rearrange('b e (h s1) (w s2) -> b (h w) (s1 s2 e)',
                          s1=patch_size, s2=patch_size),
                nn.Linear(patch_size * patch_size * in_channels, emb_size),
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(
            (img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)

        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

# establish multi-head attention


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        # queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    # Forward pass
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(
            self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        # batch, num_heads, query_len, key_len
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        # For masks
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        # apply softmax
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# Res add class


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# Feed forward class


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# Encoder block class


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

# Encoder wrapper class


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs)
                           for _ in range(depth)])

# Classification head class


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential, Base):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 32,
                 depth: int = 6,
                 num_classes: int = 1000,
                 model_name: str = 'vit',
                 model_run_no: int = 1,
                 conv_feature_extractor: bool = True,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size,
                           img_size, conv_feature_extractor,),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )
        # Class vars for classification base
        self.model_name = model_name
        self.model_run_no = model_run_no
        self.best_model_path = None
        self.best_model_state_dict = None


class PreTrained(Base):
    def __init__(self,
                 num_classes: int,
                 model_name: str = "alexnet",
                 model_run_no: int = 1,
                 ) -> None:
        super().__init__()
        # init class variables
        self.num_classes = num_classes
        self.model_name = model_name
        self.model_run_no = model_run_no
        self.best_model_path = None
        self.best_model_state_dict = None

        # load the model
        if self.model_name == 'alexnet':
            self.model = torchvision.models.alexnet(weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            print("Couldn't find model")
            return None

        # Set fc layers
        if "alexnet" in self.model_name:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.classifier[6] = nn.Linear(4096, self.num_classes)

    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x