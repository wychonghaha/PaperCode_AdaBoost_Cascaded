the dataset:
Just put the data into the floder of 'datasets/class10/test' and 'datasets/class10/train',then the classes as:
1_new Az_2S1
2_new BMP2
3_new BRDM_2
4_new BTR_60
5_new BTR70
6_new D7
7_new T62
8_new T72
9_new ZIL131
10_new ZSU_23_4

the models:
models.py #our CNN models
othermodels_all.py #the method we used to compare the acc classified by the features extracted by CNN
othermodels.py #the method we used
A_ConvNets.py #the model of A_convnet
models_concate.py # Concatenate different layers by cnn

the training:
ours_method.py #our method
ours_method_onelayer.py #try different layer by our method
Aconvnet_compare.py #Aconvnet method
all_method.py #try other ensemble method

the testing:
test1.py #used all_method model
test2.py #uesd ours_method_onelayer model
test3.py #used models_concate model
test4.py #used ours_method model

