## Objective - Reach 80% accuracy in CIFAR-10 classification problem 
## Use one dilated, one depth wise separable convolution and less than 1m params to reach the desired state

## Model
```python

Requirement already satisfied: torchsummary in c:\programdata\anaconda3\lib\site-packages (1.5.1)
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
            Conv2d-4           [-1, 64, 32, 32]          18,432
              ReLU-5           [-1, 64, 32, 32]               0
       BatchNorm2d-6           [-1, 64, 32, 32]             128
            Conv2d-7           [-1, 64, 32, 32]           4,096
         MaxPool2d-8           [-1, 64, 16, 16]               0
            Conv2d-9          [-1, 128, 12, 12]          73,728
             ReLU-10          [-1, 128, 12, 12]               0
      BatchNorm2d-11          [-1, 128, 12, 12]             256
        MaxPool2d-12            [-1, 128, 6, 6]               0
           Conv2d-13            [-1, 128, 8, 8]         147,456
             ReLU-14            [-1, 128, 8, 8]               0
      BatchNorm2d-15            [-1, 128, 8, 8]             256
        MaxPool2d-16            [-1, 128, 4, 4]               0
           Conv2d-17            [-1, 128, 4, 4]           1,280
           Conv2d-18             [-1, 64, 4, 4]           8,256
           Conv2d-19             [-1, 64, 2, 2]             640
           Conv2d-20             [-1, 32, 2, 2]           2,080
        AvgPool2d-21             [-1, 32, 1, 1]               0
           Conv2d-22             [-1, 10, 1, 1]             320
================================================================
Total params: 257,856
Trainable params: 257,856
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.56
Params size (MB): 0.98
Estimated Total Size (MB): 4.56
----------------------------------------------------------------
```
## Accuracy - 82.0% accuracy at 100th Epoch
