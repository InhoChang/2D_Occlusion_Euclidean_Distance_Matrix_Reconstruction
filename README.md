# Occlusion_EDM_2D_Reconstruction
occlusion reconstruction using 2d edm for human pose estimation. 


### Requirments 
- python 3.6 
- tensorflow-gpu 1.11 
- h5py 
- PIL 

### 1. save_img.py 
To make sparse(occlusion) EDM, randomly select joint numbers and make EDM value as zero. (num_zeros = 1, 2, 3) 
After making sparse(occlusion) EDM, save as Image file (.png) in below directory.  
image size is 16x16. label is original EDM without sparse(occlusion). 

```
img/ 
  zeros_1/ 
    tr/
     img/ 
        zeros_1_img_0 
        ...
     label/ 
        zeros_1_label_0
        ...
           
  
  
   zeros_2/ zeros_3/ 
    tr/ dev/ tt/ 
      img/
        ...
      label/ 
        ...       
   
   
```

each sparse EDM (zeros_1, zeros_2, zeros_3) has 
 tr_img : 1,247,800
 dev_img : 311,950
 tt_img : 548,818


### 2. train.py 

train and reconstruct occlusion edm using simple Fully Convolutional Network(FCN) 


model specifics:
'''
Conv - BN - Relu - Max_Pooling - Dropout x 2 
Deconv - BN - Relu - Dropout
Deconv - BN - Relu - Conv 1x1 - Relu 
L2_loss
'''

train settings are in params.json
which below in 
./experiments/base_model 


### 3. Evaluation

~ing 





