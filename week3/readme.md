###Final validation accuracy of basenetwork

```
Accuracy on test data is: 82.37

```

###Our model definition 

```
##Defineing New Model
weight_decay = 1e-4 ## https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
model1 = Sequential()
model1.add(SeparableConv2D(filters= 32,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,input_shape=(32,32,3),activation='relu'))  #OC : 30x30 RF:3
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
model1.add(SeparableConv2D(filters= 64,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #OC : 28x28 RF:5
model1.add(Dropout(0.2))
model1.add(BatchNormalization())
model1.add(SeparableConv2D(filters= 128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #OC : 26x26 RF:7
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
model1.add(MaxPooling2D(pool_size=(2, 2))) #OC : 13x13 RF:8

model1.add(SeparableConv2D(filters= 128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #OC : 11x11  RF:12
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(SeparableConv2D(filters= 256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #OC : 9x9  RF:16
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(MaxPooling2D(pool_size=(2, 2))) #OC : 4x4  RF:18



model1.add(SeparableConv2D(filters= 64,kernel_size=(1,1),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #OC : 4x4  RF:18
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
model1.add(SeparableConv2D(filters= 10,kernel_size=(1,1),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #OC : 4x4  RF:18
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
model1.add(SeparableConv2D(filters= 10,kernel_size=(4,4),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #OC : 1x1  RF:30

model1.add(GlobalAveragePooling2D())
model1.add(Activation('softmax'))

```

### Our 50 epoch log

```
usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:11: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  # This is added back by InteractiveShellApp.init_path()
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:11: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., callbacks=[<keras.ca..., steps_per_epoch=390, epochs=50)`
  # This is added back by InteractiveShellApp.init_path()
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.005.
390/390 [==============================] - 33s 84ms/step - loss: 0.7065 - acc: 0.7533 - val_loss: 0.9066 - val_acc: 0.7050
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0037907505686125857.
390/390 [==============================] - 29s 75ms/step - loss: 0.6453 - acc: 0.7757 - val_loss: 0.7563 - val_acc: 0.7371
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.003052503052503053.
390/390 [==============================] - 29s 74ms/step - loss: 0.6066 - acc: 0.7898 - val_loss: 0.6988 - val_acc: 0.7633
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0025549310168625446.
390/390 [==============================] - 29s 74ms/step - loss: 0.5793 - acc: 0.7995 - val_loss: 0.6819 - val_acc: 0.7674
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0021968365553602814.
390/390 [==============================] - 29s 74ms/step - loss: 0.5630 - acc: 0.8049 - val_loss: 0.6312 - val_acc: 0.7857
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.001926782273603083.
390/390 [==============================] - 29s 74ms/step - loss: 0.5427 - acc: 0.8115 - val_loss: 0.6412 - val_acc: 0.7846
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0017158544955387784.
390/390 [==============================] - 29s 74ms/step - loss: 0.5301 - acc: 0.8154 - val_loss: 0.6283 - val_acc: 0.7869
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0015465511908444168.
390/390 [==============================] - 29s 74ms/step - loss: 0.5177 - acc: 0.8202 - val_loss: 0.5975 - val_acc: 0.7956
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0014076576576576576.
390/390 [==============================] - 29s 74ms/step - loss: 0.5066 - acc: 0.8237 - val_loss: 0.6634 - val_acc: 0.7825
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0012916559028674762.
390/390 [==============================] - 29s 74ms/step - loss: 0.4898 - acc: 0.8301 - val_loss: 0.5789 - val_acc: 0.8023
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0011933174224343678.
390/390 [==============================] - 29s 74ms/step - loss: 0.4870 - acc: 0.8305 - val_loss: 0.5741 - val_acc: 0.8051
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0011088933244621866.
390/390 [==============================] - 29s 74ms/step - loss: 0.4784 - acc: 0.8339 - val_loss: 0.6012 - val_acc: 0.7996
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.001035625517812759.
390/390 [==============================] - 29s 75ms/step - loss: 0.4693 - acc: 0.8356 - val_loss: 0.5820 - val_acc: 0.8009
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0009714396735962696.
390/390 [==============================] - 29s 74ms/step - loss: 0.4685 - acc: 0.8372 - val_loss: 0.5817 - val_acc: 0.8026
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0009147457006952067.
390/390 [==============================] - 29s 74ms/step - loss: 0.4593 - acc: 0.8404 - val_loss: 0.5758 - val_acc: 0.8077
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.000864304235090752.
390/390 [==============================] - 29s 74ms/step - loss: 0.4549 - acc: 0.8404 - val_loss: 0.5683 - val_acc: 0.8043
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00081913499344692.
390/390 [==============================] - 29s 74ms/step - loss: 0.4485 - acc: 0.8407 - val_loss: 0.5661 - val_acc: 0.8129
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0007784524365561264.
390/390 [==============================] - 29s 74ms/step - loss: 0.4500 - acc: 0.8428 - val_loss: 0.5427 - val_acc: 0.8133
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0007416196974191634.
390/390 [==============================] - 29s 74ms/step - loss: 0.4339 - acc: 0.8480 - val_loss: 0.5813 - val_acc: 0.8064
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0007081149978756551.
390/390 [==============================] - 29s 74ms/step - loss: 0.4355 - acc: 0.8478 - val_loss: 0.5533 - val_acc: 0.8163
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0006775067750677507.
390/390 [==============================] - 29s 74ms/step - loss: 0.4336 - acc: 0.8482 - val_loss: 0.5425 - val_acc: 0.8173
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0006494349915573452.
390/390 [==============================] - 29s 74ms/step - loss: 0.4262 - acc: 0.8496 - val_loss: 0.5416 - val_acc: 0.8169
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0006235969069593414.
390/390 [==============================] - 29s 74ms/step - loss: 0.4229 - acc: 0.8513 - val_loss: 0.5853 - val_acc: 0.8046
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0005997361161089121.
390/390 [==============================] - 29s 73ms/step - loss: 0.4231 - acc: 0.8531 - val_loss: 0.5480 - val_acc: 0.8175
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.000577634011090573.
390/390 [==============================] - 28s 72ms/step - loss: 0.4225 - acc: 0.8534 - val_loss: 0.5286 - val_acc: 0.8225
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0005571030640668523.
390/390 [==============================] - 28s 73ms/step - loss: 0.4133 - acc: 0.8571 - val_loss: 0.5164 - val_acc: 0.8244
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0005379814934366257.
390/390 [==============================] - 28s 73ms/step - loss: 0.4185 - acc: 0.8540 - val_loss: 0.5315 - val_acc: 0.8233
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0005201289919900136.
390/390 [==============================] - 28s 73ms/step - loss: 0.4108 - acc: 0.8562 - val_loss: 0.5188 - val_acc: 0.8266
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0005034232782923882.
390/390 [==============================] - 28s 73ms/step - loss: 0.4101 - acc: 0.8558 - val_loss: 0.5241 - val_acc: 0.8225
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.000487757291971515.
390/390 [==============================] - 28s 73ms/step - loss: 0.4020 - acc: 0.8590 - val_loss: 0.5194 - val_acc: 0.8283
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0004730368968779565.
390/390 [==============================] - 28s 72ms/step - loss: 0.3987 - acc: 0.8589 - val_loss: 0.5178 - val_acc: 0.8258
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.00045917898796951054.
390/390 [==============================] - 28s 73ms/step - loss: 0.4046 - acc: 0.8579 - val_loss: 0.5089 - val_acc: 0.8291
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0004461099214846538.
390/390 [==============================] - 28s 73ms/step - loss: 0.4001 - acc: 0.8599 - val_loss: 0.5229 - val_acc: 0.8235
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0004337642057777392.
390/390 [==============================] - 28s 73ms/step - loss: 0.4005 - acc: 0.8593 - val_loss: 0.5359 - val_acc: 0.8192
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0004220834036805673.
390/390 [==============================] - 29s 73ms/step - loss: 0.3983 - acc: 0.8598 - val_loss: 0.5279 - val_acc: 0.8235
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0004110152075626798.
390/390 [==============================] - 28s 73ms/step - loss: 0.3952 - acc: 0.8612 - val_loss: 0.5156 - val_acc: 0.8249
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0004005126561999359.
390/390 [==============================] - 28s 73ms/step - loss: 0.4005 - acc: 0.8607 - val_loss: 0.5140 - val_acc: 0.8268
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.00039053346871826915.
390/390 [==============================] - 28s 73ms/step - loss: 0.3938 - acc: 0.8608 - val_loss: 0.5442 - val_acc: 0.8165
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.00038103947568968145.
390/390 [==============================] - 28s 73ms/step - loss: 0.3908 - acc: 0.8626 - val_loss: 0.5157 - val_acc: 0.8273
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0003719961312402351.
390/390 [==============================] - 29s 73ms/step - loss: 0.3899 - acc: 0.8628 - val_loss: 0.5194 - val_acc: 0.8251
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0003633720930232558.
390/390 [==============================] - 28s 73ms/step - loss: 0.3919 - acc: 0.8627 - val_loss: 0.5312 - val_acc: 0.8231
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.00035513885929398396.
390/390 [==============================] - 29s 73ms/step - loss: 0.3873 - acc: 0.8626 - val_loss: 0.5185 - val_acc: 0.8291
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0003472704542297541.
390/390 [==============================] - 28s 73ms/step - loss: 0.3851 - acc: 0.8656 - val_loss: 0.5124 - val_acc: 0.8298
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.00033974315417544334.
390/390 [==============================] - 28s 73ms/step - loss: 0.3820 - acc: 0.8641 - val_loss: 0.5331 - val_acc: 0.8243
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.00033253524873636604.
390/390 [==============================] - 29s 73ms/step - loss: 0.3837 - acc: 0.8666 - val_loss: 0.5334 - val_acc: 0.8214
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.00032562683165092806.
390/390 [==============================] - 29s 73ms/step - loss: 0.3861 - acc: 0.8645 - val_loss: 0.5154 - val_acc: 0.8292
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0003189996172004594.
390/390 [==============================] - 28s 73ms/step - loss: 0.3823 - acc: 0.8643 - val_loss: 0.5187 - val_acc: 0.8276
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0003126367785906334.
390/390 [==============================] - 29s 73ms/step - loss: 0.3787 - acc: 0.8659 - val_loss: 0.5372 - val_acc: 0.8217
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.00030652280529671407.
390/390 [==============================] - 29s 73ms/step - loss: 0.3785 - acc: 0.8669 - val_loss: 0.5215 - val_acc: 0.8270
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0003006433768264085.
390/390 [==============================] - 28s 73ms/step - loss: 0.3759 - acc: 0.8689 - val_loss: 0.5175 - val_acc: 0.8270
Model took 1435.62 seconds to train

Accuracy on test data is: 82.70

```

###team
Nitin(https://github.com/nitinranjansharma)
Rahul(https://github.com/forsc)
Nilanjan(https://github.com/Nildk123)


