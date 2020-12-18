## S4 -EVA
Reach 99.4% Validation accuracy on MNIST with less than 20k Parameter & less than 20 Epoch

### Method/Model used to reach
Model with an maximum of 16 kernel with batchnorm is used, full model description & Code block with RF calculation is given beloow

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1,bias = False)  #input -28*28*1 OUtput - 28*28*16 RF - 3x3
        self.batchnorm1 = nn.BatchNorm2d(16)                      #input -28*28*16 OUtput - 28*28*16 RF - 3x3
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1,bias = False) #input -28*28*16 OUtput - 28*28*16 RF - 5x5
        self.batchnorm2 = nn.BatchNorm2d(16)                      #input -28*28*16 OUtput - 28*28*16 RF - 5x5
        self.pool1 = nn.MaxPool2d(2, 2)                           #input -28*28*16 OUtput - 14*14*16 RF - 10x10  
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1,bias = False) #input - 14*14*16 OUtput - 14*14*16 RF - 12x12
        self.batchnorm3 = nn.BatchNorm2d(16)                      #input - 14*14*16 OUtput - 14*14*16 RF - 12x12
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1,bias = False) #input - 14*14*16 OUtput - 14*14*16 RF - 14x14
        self.batchnorm4 = nn.BatchNorm2d(16)                      #input - 14*14*16 OUtput - 14*14*16 RF - 14x14
        self.pool2 = nn.MaxPool2d(2, 2)                           #inpit - 14*14*16 output - 7*7*16   RF - 28*28
        self.conv5 = nn.Conv2d(16, 16, 3,bias = False)            #input - 7*7*16 output - 5*5*16 RF - 30*30
        self.batchnorm5 = nn.BatchNorm2d(16)                      #input - 5*5*16 output - 5*5*16 RF - 30*30  
        self.conv6 = nn.Conv2d(16, 10, 3,bias = False)            #input - 5*5*16 output - 3*3*10 RF - 32*32
        
        

    def forward(self, x):
        x = self.pool1(F.relu(self.batchnorm2(self.conv2(self.batchnorm1(F.relu(self.conv1(x)))))))
        x = self.pool2(self.batchnorm4(F.relu(self.conv4(self.batchnorm3(F.relu(self.conv3(x)))))))
        x = F.relu(self.conv6(self.batchnorm5(F.relu(self.conv5(x))))) #Final RF 32*32
        x = F.adaptive_avg_pool2d(x, (1, 1)) #Global Avarage pooling
        x = x.view(-1, 10)
        return F.log_softmax(x)
```

### Total Parameter Layer wise 
Total number of parameter is 10960, Model summary snippet is given below for better understanding
```python
Requirement already satisfied: torchsummary in /usr/local/lib/python3.6/dist-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             144
       BatchNorm2d-2           [-1, 16, 28, 28]              32
            Conv2d-3           [-1, 16, 28, 28]           2,304
       BatchNorm2d-4           [-1, 16, 28, 28]              32
         MaxPool2d-5           [-1, 16, 14, 14]               0
            Conv2d-6           [-1, 16, 14, 14]           2,304
       BatchNorm2d-7           [-1, 16, 14, 14]              32
            Conv2d-8           [-1, 16, 14, 14]           2,304
       BatchNorm2d-9           [-1, 16, 14, 14]              32
        MaxPool2d-10             [-1, 16, 7, 7]               0
           Conv2d-11             [-1, 16, 5, 5]           2,304
      BatchNorm2d-12             [-1, 16, 5, 5]              32
           Conv2d-13             [-1, 10, 3, 3]           1,440
================================================================
Total params: 10,960
Trainable params: 10,960
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.52
Params size (MB): 0.04
Estimated Total Size (MB): 0.56
----------------------------------------------------------------
```
### Validation log

We reached a maximum validation accuracy 99.44% at Epoch 10.
Full validation accuracy of 20 epoch is given below

```python
 0%|          | 0/1875 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:26: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.02192363142967224 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 85.35it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0389, Accuracy: 9886/10000 (99%)

loss=0.0018165125511586666 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 85.95it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0419, Accuracy: 9868/10000 (99%)

loss=0.012154631316661835 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 85.97it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0301, Accuracy: 9899/10000 (99%)

loss=0.006491743493825197 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.03it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0268, Accuracy: 9915/10000 (99%)

loss=0.01395842619240284 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 85.77it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0286, Accuracy: 9907/10000 (99%)

loss=0.004266486968845129 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.35it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0226, Accuracy: 9931/10000 (99%)

loss=0.025564849376678467 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.60it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0252, Accuracy: 9914/10000 (99%)

loss=0.03266138210892677 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 85.69it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0214, Accuracy: 9931/10000 (99%)

loss=0.00024779970408417284 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 85.84it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0194, Accuracy: 9935/10000 (99%)

loss=0.0027017013635486364 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 87.17it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9944/10000 (99%)

loss=0.0499432235956192 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.82it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9939/10000 (99%)

loss=0.0020160318817943335 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 87.01it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0228, Accuracy: 9927/10000 (99%)

loss=0.0015100210439413786 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 87.22it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0237, Accuracy: 9927/10000 (99%)

loss=9.928153303917497e-05 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.23it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0208, Accuracy: 9936/10000 (99%)

loss=0.000304381683235988 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.03it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0190, Accuracy: 9943/10000 (99%)

loss=0.0001533225440653041 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 87.27it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0212, Accuracy: 9943/10000 (99%)

loss=6.201746145961806e-05 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.30it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0214, Accuracy: 9935/10000 (99%)

loss=0.004394373390823603 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 87.02it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9940/10000 (99%)

loss=0.0007066067773848772 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.92it/s]

Test set: Average loss: 0.0233, Accuracy: 9939/10000 (99%)

```



