# 2D_Occlusion_Euclidean_Distance_Matrix_Reconstruction
Occlusion reconstruction using simple fully convolution layer network 
with 2d euclidean distance matrix(edm) for human pose estimation.
This code is starting from making occluded edm and reconstruct edm, so you have to prepare EDM dataset from human3.6m using 2d joint coordinates. 

# Update 19.05.10
apply min_max normalization to logits. 
previous logit values range was [-2 8]. However, the ground turth(label) was [0 1]. 
so match both values range between [0 1] using min_max normalization.

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
```
Conv - BN - Relu - Max_Pooling - Dropout x 2 

Deconv - BN - Relu - Dropout

Deconv - BN - Relu - Conv 1x1 - Relu 

L2_loss
```

train settings are in params.json
which below in 
./experiments/base_model 


### 3. Evaluation

~ing 





