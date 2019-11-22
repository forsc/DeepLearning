### Strategy 
we saw that base network performing wonderfully with little high number of parameters (reaching accuracies 0.9947).

```
=================================================================
Total params: 16,604
Trainable params: 16,360
Non-trainable params: 244
_________________________________________________________________
```
So our stragtegy was to just tweak the channels in the intial layers to a lower amount and keep the network as it is. And it worked almost same as the base network

```
=================================================================
Total params: 12,516
Trainable params: 12,316
Non-trainable params: 200
_________________________________________________________________

```

As seen with the parameter summary we reached a test accurecy of .9945. with 4k less parameters

Our test accurecy 

```
[0.0208969491261465, 0.9945]
```

And the Epoch log

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 24s 398us/step - loss: 0.0457 - acc: 0.9862 - val_loss: 0.0378 - val_acc: 0.9894
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.002274450341167551.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0374 - acc: 0.9888 - val_loss: 0.0340 - val_acc: 0.9910
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018315018317.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0347 - acc: 0.9895 - val_loss: 0.0301 - val_acc: 0.9904
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586101175269.
60000/60000 [==============================] - 7s 110us/step - loss: 0.0323 - acc: 0.9905 - val_loss: 0.0287 - val_acc: 0.9926
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.001318101933216169.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0284 - acc: 0.9916 - val_loss: 0.0256 - val_acc: 0.9937
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560693641618498.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0266 - acc: 0.9925 - val_loss: 0.0214 - val_acc: 0.9939
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.001029512697323267.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0247 - acc: 0.9926 - val_loss: 0.0241 - val_acc: 0.9933
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307145066501.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0243 - acc: 0.9927 - val_loss: 0.0220 - val_acc: 0.9944
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445945945945946.
60000/60000 [==============================] - 7s 113us/step - loss: 0.0227 - acc: 0.9933 - val_loss: 0.0224 - val_acc: 0.9939
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935417204857.
60000/60000 [==============================] - 7s 110us/step - loss: 0.0235 - acc: 0.9930 - val_loss: 0.0229 - val_acc: 0.9942
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159904534606206.
60000/60000 [==============================] - 7s 110us/step - loss: 0.0223 - acc: 0.9934 - val_loss: 0.0233 - val_acc: 0.9942
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0006653359946773121.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0214 - acc: 0.9938 - val_loss: 0.0221 - val_acc: 0.9943
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753106876554.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0219 - acc: 0.9938 - val_loss: 0.0236 - val_acc: 0.9940
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638041577617.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0202 - acc: 0.9941 - val_loss: 0.0216 - val_acc: 0.9942
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474204171241.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0192 - acc: 0.9945 - val_loss: 0.0208 - val_acc: 0.9947
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825410544512.
60000/60000 [==============================] - 7s 110us/step - loss: 0.0204 - acc: 0.9942 - val_loss: 0.0217 - val_acc: 0.9944
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491480996068152.
60000/60000 [==============================] - 7s 112us/step - loss: 0.0193 - acc: 0.9943 - val_loss: 0.0228 - val_acc: 0.9944
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670714619336759.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0191 - acc: 0.9946 - val_loss: 0.0222 - val_acc: 0.9943
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718184514981.
60000/60000 [==============================] - 7s 112us/step - loss: 0.0182 - acc: 0.9946 - val_loss: 0.0209 - val_acc: 0.9942
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.00042486899872539303.
60000/60000 [==============================] - 7s 111us/step - loss: 0.0188 - acc: 0.9942 - val_loss: 0.0209 - val_acc: 0.9945
<keras.callbacks.History at 0x7f4e4aec63c8>


```

#### Team 
Nitin(https://github.com/nitinranjansharma)
Rahul(https://github.com/forsc)
Nilanjan(https://github.com/Nildk123)
