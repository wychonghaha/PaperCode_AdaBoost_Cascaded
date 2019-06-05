from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import os
from A_ConvNets import *
from othermodels import all_model
from read_images import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.models import Model


mode='test' #测试还是训练 train 训练网络 test 测试网络 pre 其他为测试网络及机器学习模型
test_numbers=10 #一共测多少次取平均值
part_or_all='part' #样本数 all 为取全部样本 其他需要改 total 参数
total=0#取样本数
suiji_or_chazhi='' #取模型的名字
suiji=False #是随机取还是隔一定角度取

num_classes =10  #类别数
mode_concate='max' #结合方式 
img_size = (88, 88)
epochs =100
batch_size =32

def change_shape(data,type='min'):
    if type=='max':
        fx=np.max(data,axis=-1)
    elif type=='average':
        fx=np.average(data,axis=-1)
    elif type=='no':
        fx=data
    else:
        fx=np.min(data,axis=-1)
    fd=np.reshape(fx,(fx.shape[0],-1))
    return fd

datasets=[5,10,20,30,40,50]

for i in datasets:
    total=i
    #若已经生成了数据，则直接加载
    x_train,y_train,x_test, y_test=get_alldata(num_classes=num_classes,img_size=img_size) #第一遍取时遍历所有的样本文件夹，然后根据num_classes取前几个文件夹作为训练及测试
    if part_or_all!='all':
        x_train,y_train=get_partdata(x_train,y_train,num_classes=num_classes,total=total,mode='train',suiji=suiji)
    x_train=x_train[:,20:108,20:108,:]
    x_test=x_test[:,20:108,20:108,:]
    #加载网络模型
    model_path='model_aconv/deep_model_'+part_or_all+str(total)+'.h5'#+'_'+str(num_classes) +'_'+suiji_or_chazhi
    model=AconvNet(model_path,num_classes=num_classes)
    if mode=='train':
        checkpoint = ModelCheckpoint(filepath=model_path,monitor='val_acc',mode='auto' ,save_best_only='True')
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),callbacks=[checkpoint])
        s=model.evaluate(x_test,y_test)
        print('CNN:',s)
    elif mode=='test':
        s=model.evaluate(x_test,y_test)
        print(str(total)+' samples/one_category test_acc:',s)
    else:
         #Aconv_net
        model_path_aconv='model_aconv/deep_model_'+part_or_all+str(total)+'.h5'#+'_'+str(num_classes) +'_'+suiji_or_chazhi
        model_aconv=AconvNet(model_path_aconv,num_classes=num_classes)
        result_total=[]
        for i in range(test_numbers):
            #第3层
            dense1_layer_model =Model(inputs=model_aconv.input,outputs=model_aconv.get_layer('layer1').output)  
            fx_test3 = dense1_layer_model.predict(x_test)
            fx_train3 = dense1_layer_model.predict(x_train)
            fx_test3=change_shape(fx_test3,type='no')
            fx_train3=change_shape(fx_train3,type='no')
            pca=PCA(n_components=128)
            pca.fit(fx_train3)
            fx_tr3=pca.transform(fx_train3)
            fx_te3=pca.transform(fx_test3)
            
            #第4层
            dense1_layer_model =Model(inputs=model_aconv.input,outputs=model_aconv.get_layer('layer2').output)  
            fx_test4 = dense1_layer_model.predict(x_test)
            fx_train4 = dense1_layer_model.predict(x_train)
            fx_test4=change_shape(fx_test4,type='no')
            fx_train4=change_shape(fx_train4,type='no')
            pca=PCA(n_components=86)
            pca.fit(fx_train4)
            fx_tr4=pca.transform(fx_train4)
            fx_te4=pca.transform(fx_test4)      
            
            #第五层
            dense1_layer_model =Model(inputs=model_aconv.input,outputs=model_aconv.get_layer('layer3').output)  
            fx_test_dence1 = dense1_layer_model.predict(x_test)
            fx_train_dence1 = dense1_layer_model.predict(x_train)
            fx_test_dence1=change_shape(fx_test_dence1,type='no')
            fx_train_dence1=change_shape(fx_train_dence1,type='no')

            #第六层
            dense1_layer_model_2 =Model(inputs=model_aconv.input,outputs=model_aconv.get_layer('layer4').output)  
            fx_test_out = dense1_layer_model_2.predict(x_test)
            fx_train_out = dense1_layer_model_2.predict(x_train)
            fx_test_out=change_shape(fx_test_out,type='no')
            fx_train_out=change_shape(fx_train_out,type='no')

            fx_train=np.concatenate((fx_tr3,fx_tr4,fx_train_dence1,fx_train_out),axis=1)  
            fx_test=np.concatenate((fx_te3,fx_te4,fx_test_dence1,fx_test_out),axis=1) 
            
            y_train1=np.argmax(y_train,axis=1)
            y_test1=np.argmax(y_test,axis=1)
            result=all_model(fx_train,fx_test,y_train1,y_test1)
            result_total.append(result)
            print(i,'Rotation Forest:',result[0],'Adaboost RF:',result[1])
        # s1=model.evaluate(x_test,y_test)
        # x_test=x_test1[:,20:108,20:108,:]
        s2=model_aconv.evaluate(x_test,y_test)
        result_total=np.array(result_total)
        d=np.max(result_total,0)
        print(str(total)+' samples/one_category test_acc:','A_convNet:',s2[1],'Rotation Forest:',d[0],'Adaboost RF:',d[1])




