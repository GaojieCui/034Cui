# 深度学习基础

完整的深度学习训练套路

![image.png](attachment:42b0a0a2-895c-4b83-85c3-268d79fdf5d5:image.png)

训练一定是两次循环

欠拟合：训练训练数据集表现不好，验证表现不好

过拟合：训练数据训练过程表现得很好，在我得验证过程表现不好

![image.png](attachment:c9f9e8ed-69b4-4830-88fa-023048678d72:image.png)

# 卷积神经网络

卷积过程

```jsx
import torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# 不满足conv2d的尺寸要求
print(input.shape)
print(kernel.shape)

# 尺寸变换
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
print(input.shape)
print(kernel.shape)

output = F.conv2d(input=input,weight=kernel,stride=1)
print(output)

output2 = F.conv2d(input=input,weight=kernel,stride=2)
print(output2)

# padding 在周围扩展一个像素，默认为0；
output3 = F.conv2d(input=input,weight=kernel,stride=1,padding=1)
print(output3)
```

5*5的输入数据 3*3的卷积核 步长1 填充1，

## 

### 图片卷积

```jsx
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

class CHEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=3,
                               stride=1,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

chen = CHEN()
print(chen)

writer = SummaryWriter("conv_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = chen(imgs)

    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # print(output.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) ->([**, 3, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))  # -1:会根据后面的值进行调整
    writer.add_images("output", output, step)
    step += 1

定义我们的网络模型
```

![image.png](attachment:59df05a7-f114-4c4a-95c5-be72ea153ecc:image.png)

### tensorboard使用

使用之前安装一下tensorboard

这段代码的作用只是为了拿到我的conv_logs里面的文件

使用tensorboard命令打开

tensorboard --logdir=conv_logs

![image.png](attachment:c589f4aa-51e5-4e67-8c5b-cba9a683eb8a:image.png)

点击链接得到一下界面

![image.png](attachment:617ee179-818a-462e-ac06-36df203e73a7:image.png)

# 池化层

代码里面是最大池化，还有平均池化

```jsx
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#
dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

# # 最大池化没法对long整形进行池化
# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]], dtype = torch.float)
# input =torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_1 = MaxPool2d(kernel_size=3,
                                   ceil_mode=False)
    def forward(self,input):
        output = self.maxpool_1(input)
        return output

chen = Chen()

writer = SummaryWriter("maxpool_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output = chen(imgs)
    writer.add_images("ouput",output,step)
    step += 1
writer.close()

#
# output = chen(input)
# print(output)
```

![image.png](attachment:522a3517-3329-48c1-a6ad-6453dd710605:image.png)

池化后的维度

![image.png](attachment:7a568c93-3e2d-4761-a257-d0cab3ee02b0:image.png)

# 作业：搭建alexnet

![image.png](attachment:94341c4a-913a-4525-a512-e9a83392ebdc:image.png)

alexnet 如下

```python
import torch
from torch import nn

class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=4),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(48, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 192, kernel_size=3),
            nn.Conv2d(192, 192, kernel_size=3),
            nn.Conv2d(192, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        y = self.model(x)

        return y

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    alexnet = alex()
    y = alexnet(x)
    print(y.shape)
```

### 尝试使用resnet，Googlenet，mobileNet，moganet 等不同模型跑图片分类

### 尝试使用GPU训练网络模型
