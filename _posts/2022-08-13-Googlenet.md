---
title:  "GoogleNet pytorch 구현"
excerpt: "torch, cifar10"
categories:
  - Deep learning

toc: true
toc_sticky: true
---
# GoogleNet

2014년에 발표한 논문, 당시 image classification task 1등을 차지.
Inception Block, Auxiliary Classifier 등 이용.

논문을 읽고 torchvision 모델을 살펴보면서 파이토치로 구현해 봤다.

## 필요 라이브러리
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
```

## 모델 하이퍼 파라미터 정리

깊은 모델이고 하이퍼 파라미터가 다 다르기 때문에 딕셔너리 형태로 따로 정리했다.
```python
# 분류 모델 아웃풋 개수 지정.
n_out = 10

params = {
    # conv: in_channels, out_channels, kernel_size, stride, padding
                                            # featuremap size
    'conv1':[3, 64, 7, 2, 3],               # 112
    # maxpool: kernel, stride, padding             
    'maxpool1':[3, 2, 1],                   # 56
    'conv2':[64, 64, 1, 1],
    'conv3': [64, 192, 3, 1, 1],            # 56
    'maxpool2': [3, 2, 1],                  # 28
    

                                                            # block size
    'inception_3a': {
        'branch1' : [192, 64, 1, 1],                        # 64
        'branch2' : {'conv1': [192, 96, 1, 1],
                     'conv3': [96, 128, 3, 1, 1]},          # 64 + 128 = 192
                     
        'branch3' : {'conv1': [192, 16, 1, 1],
                     'conv3': [16, 32, 3, 1, 1]},           # 192 + 32 = 224
                    
        'branch4' : {'maxpool': [3, 1, 1],
                     'conv1': [192, 32, 1, 1]}              # 224 + 32 = 256
    },

    'inception_3b': {
        'branch1' : [256, 128, 1, 1],                        # 64
        'branch2' : {'conv1': [256, 128, 1, 1],
                     'conv3': [128, 192, 3, 1, 1]},          # 64 + 128 = 192
                     
        'branch3' : {'conv1': [256, 32, 1, 1],
                     'conv3': [32, 96, 3, 1, 1]},           # 192 + 32 = 224
                    
        'branch4' : {'maxpool': [3, 1, 1],
                     'conv1': [256, 64, 1, 1]} 
    },

    # feature map : 14
    'maxpool3': [3, 2, 1],


    'inception_4a': {
        'branch1' : [480, 192, 1, 1],                        # 64
        'branch2' : {'conv1': [480, 96, 1, 1],
                     'conv3': [96, 208, 3, 1, 1]},          # 64 + 128 = 192
                     
        'branch3' : {'conv1': [480, 16, 1, 1],
                     'conv3': [16, 48, 3, 1, 1]},           # 192 + 32 = 224
                    
        'branch4' : {'maxpool': [3, 1, 1],
                     'conv1': [480, 64, 1, 1]} 
    },

    'aux1':{
        'avgpool':[5, 3],
        'conv' : [512, 128, 1, 1],
        'fc1' : [2048, 1024],
        'fc2' : [1024, n_out]
    },
    
    'inception_4b': {
        'branch1' : [512, 160, 1, 1],                        # 64
        'branch2' : {'conv1': [512, 112, 1, 1],
                     'conv3': [112, 224, 3, 1, 1]},          # 64 + 128 = 192
                     
        'branch3' : {'conv1': [512, 24, 1, 1],
                     'conv3': [24, 64, 3, 1, 1]},           # 192 + 32 = 224
                    
        'branch4' : {'maxpool': [3, 1, 1],
                     'conv1': [512, 64, 1, 1]} 
    },

    'inception_4c': {
        'branch1' : [512, 128, 1, 1],                        # 64
        'branch2' : {'conv1': [512, 128, 1, 1],
                     'conv3': [128, 256, 3, 1, 1]},          # 64 + 128 = 192
                     
        'branch3' : {'conv1': [512, 24, 1, 1],
                     'conv3': [24, 64, 3, 1, 1]},           # 192 + 32 = 224
                    
        'branch4' : {'maxpool': [3, 1, 1],
                     'conv1': [512, 64, 1, 1]} 
    },

    'inception_4d': {
        'branch1' : [512, 112, 1, 1],                        # 64
        'branch2' : {'conv1': [512, 144, 1, 1],
                     'conv3': [144, 288, 3, 1, 1]},          # 64 + 128 = 192
                     
        'branch3' : {'conv1': [512, 32, 1, 1],
                     'conv3': [32, 64, 3, 1, 1]},           # 192 + 32 = 224
                    
        'branch4' : {'maxpool': [3, 1, 1],
                     'conv1': [512, 64, 1, 1]} 
    },

    'aux2':{
        'avgpool':[5, 3],
        'conv' : [528, 128, 1, 1],
        'fc1' : [2048, 1024],
        'fc2' : [1024, n_out]
    },

    'inception_4e': {
        'branch1' : [528, 256, 1, 1],                        # 64
        'branch2' : {'conv1': [528, 160, 1, 1],
                     'conv3': [160, 320, 3, 1, 1]},          # 64 + 128 = 192
                     
        'branch3' : {'conv1': [528, 32, 1, 1],
                     'conv3': [32, 128, 3, 1, 1]},           # 192 + 32 = 224
                    
        'branch4' : {'maxpool': [3, 1, 1],
                     'conv1': [528, 128, 1, 1]} 
    },
    'maxpool4': [3, 2, 1],

    
    'inception_5a': {
        'branch1' : [832, 256, 1, 1],                        # 64
        'branch2' : {'conv1': [832, 160, 1, 1],
                     'conv3': [160, 320, 3, 1, 1]},          # 64 + 128 = 192
                     
        'branch3' : {'conv1': [832, 32, 1, 1],
                     'conv3': [32, 128, 3, 1, 1]},           # 192 + 32 = 224
                    
        'branch4' : {'maxpool': [3, 1, 1],
                     'conv1': [832, 128, 1, 1]} 
    },

    'inception_5b': {
        'branch1' : [832, 384, 1, 1],                        # 64
        'branch2' : {'conv1': [832, 192, 1, 1],
                     'conv3': [192, 384, 3, 1, 1]},          # 64 + 128 = 192
                     
        'branch3' : {'conv1': [832, 48, 1, 1],
                     'conv3': [48, 128, 3, 1, 1]},           # 192 + 32 = 224
                    
        'branch4' : {'maxpool': [3, 1, 1],
                     'conv1': [832, 128, 1, 1]} 
    },

    'avgpool' : [(1,1)],
    'dropout' : [0.2],
    'fc' : [1024, n_out]

}
```

## Block 구현
BasicConv Block: 컨볼루션 레이어 + 배치 정규화  
Inception Block: 4개의 branch   
Aux Block

```python
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
            out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.branch1 = BasicConv2d(*kwargs['branch1'])
        self.branch2 = nn.Sequential(
            BasicConv2d(*kwargs['branch2']['conv1']),
            BasicConv2d(*kwargs['branch2']['conv3'])
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(*kwargs['branch3']['conv1']),
            BasicConv2d(*kwargs['branch3']['conv3'])
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(*kwargs['branch4']['maxpool']),
            BasicConv2d(*kwargs['branch4']['conv1'])
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat((x1, x2, x3, x4), dim=1)
        
class InceptionAux(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.avgpool = nn.AvgPool2d(*kwargs['avgpool'])
        self.conv = BasicConv2d(*kwargs['conv'])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(*kwargs['fc1'], bias=True)
        self.fc2 = nn.Linear(*kwargs['fc2'], bias=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```
## 모델 구현
위에서 만들어둔 하이퍼 파라미터 딕셔너리를 parameter로 받는 상황으로 모델을 작성했다.
```python 
class GoogleNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1 = BasicConv2d(*params['conv1'])
        self.maxpool1 = nn.MaxPool2d(*params['maxpool1'])
        self.conv2 = BasicConv2d(*params['conv2'])
        self.conv3 = BasicConv2d(*params['conv3'])
        self.maxpool2 = nn.MaxPool2d(*params['maxpool2'])

        self.inception_3a = InceptionBlock(**params['inception_3a'])
        self.inception_3b = InceptionBlock(**params['inception_3b'])
        self.maxpool3 = nn.MaxPool2d(*params['maxpool3'])
        
        self.inception_4a = InceptionBlock(**params['inception_4a'])
        self.aux1 = InceptionAux(**params['aux1'])
        self.inception_4b = InceptionBlock(**params['inception_4b'])
        self.inception_4c = InceptionBlock(**params['inception_4c'])
        self.inception_4d = InceptionBlock(**params['inception_4d'])
        self.aux2 = InceptionAux(**params['aux2'])
        self.inception_4e = InceptionBlock(**params['inception_4e'])
        self.maxpool4 = nn.MaxPool2d(*params['maxpool4'])
        
        self.inception_5a = InceptionBlock(**params['inception_5a'])
        self.inception_5b = InceptionBlock(**params['inception_5b'])
        self.avgpool = nn.AdaptiveAvgPool2d(*params['avgpool'])
        self.dropout = nn.Dropout(*params['dropout'])
        self.fc = nn.Linear(*params['fc'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool3(x)

        inception_4a_out = self.inception_4a(x)
        
        x = self.inception_4b(inception_4a_out)
        x = self.inception_4c(x)

        inception_4d_out  = self.inception_4d(x)
    
        x = self.inception_4e(inception_4d_out)
        x = self.maxpool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x= self.fc(x)

        if self.training:
            aux1 = self.aux1(inception_4a_out)
            aux2 = self.aux2(inception_4d_out)

            return x, aux1, aux2
        else:
            return x
```

## Training Loop 
Aux Classifier가 있는 경우와 없는 경우를 구분해서 작성.
```python
def train_loop(model, device, loss_fn, optimizer, train_loader, valid_loader, n_epochs=100, aux_logit=False):
    
    for epoch in range(1, n_epochs+1):   
        model.train()
        train_epoch_loss = 0 

        for img, label in train_loader:
            img = img.to(device)
            label = torch.LongTensor(label).to(device)
            
            if aux_logit:
                out, aux1, aux2 = model(img)
                loss = loss_fn(out, label)
                aux_loss1 = loss_fn(aux1, label)
                aux_loss2 = loss_fn(aux2, label)
                loss = loss + aux_loss1 + aux_loss2
            
            else:
                out = model(img)
                loss = loss_fn(out, label)

            optimizer.zero_grad()
            # using CrossEntropyLoss
            train_epoch_loss += loss

            loss.backward()
            optimizer.step()

        
        if epoch % 1 == 0:
            valid_epoch_loss = 0
            model.eval()
            with torch.no_grad():
                for img, label in valid_loader:
                    img = img.to(device)
                    
                    label = torch.LongTensor(label).to(device)
                    out = model(img)
                    loss = loss_fn(out, label)
                    valid_epoch_loss += loss


        train_epoch_loss /= len(train_loader)
        valid_epoch_loss /= len(valid_loader)

        print(f'Epoch {epoch} Train Loss {train_epoch_loss:.4f} Valid Loss {valid_epoch_loss:.4f}')
                    
    return model
```

## Cifar Dataset 준비와 트레이닝 시작

```python
my_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()]
)

cifar_train = datasets.CIFAR10(root='data',train=True, download=True, transform=my_transform)
cifar_val = datasets.CIFAR10(root='data',train=False, download=True, transform=my_transform)

train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True)
valid_loader = DataLoader(cifar_val, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GoogleNet(params)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

model = train_loop(model, device, loss_fn, optimizer, train_loader, valid_loader, n_epochs=100, aux_logit=True)
```