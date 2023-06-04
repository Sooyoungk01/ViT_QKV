# Accelerating Vision Transformer on Jetson Nano

In this project, we try to acclerate Vision Transformer(a.k.a ViT) on Jetson Nano Developer's Kit.

**ViT** is an transformer based architecture for computer vision tasks such as image classification or object detection. In the paper, the author converts an image into 16x16 patches by patch embedding and applies self-attention based transformer encoders to find relationships between visual concepts of the patches. The ViT achieves SOTA performance on image classification tasks. You can see the original paper at [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy)

We refered to @ShivamRajSharma's PyTorch implementation of ViT.  

<p align="center">
  <img src="https://github.com/ShivamRajSharma/Vision-Transformer/blob/master/ViT.png" height="300"/>
</p>

The project was developed and tested with the following system:

- Jetson Nano Developer's Kit    
- JetPack 4.6.3           
- CUDA 10.2      
- Python 3.6      
      
<br/>     
     
## Acceleration Method: QKV Dimension Reduction  

<p align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FboPfZG%2FbtsiuAiWLUU%2F7NfijcR8xYmo4Tz11mxywk%2Fimg.png" height="300"/>
</p>  
By decreasing the dimension of input matrices from 4 to 3, the latency of calculating QKV matrices decreased to about 16%.    
 
<br/>  
 
## Train and Predict
1. Clone **ViT_QKV** repository:
  ```sh
  git clone https://github.com/Sooyoungk01/ViT_QKV.git
  ```

2. Install all the required libraries:
  ```sh
  pip install -r requirements.txt
  ```

3. Download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place it inside the cloned repository.  

4. Train:
  ```sh
  python train.py
  ```

5. Inference:
  ```sh
  python predict.py
  ```
  
<br/>

## Profiling with Nsight Systems  
For profiling, you need to install the ***nsys*** commnad from the Nsight Systems CLI. Thankfully, it is included in the JetPack 4.6.3.  
        
                
1. Create a shell script file: 
  ```sh
  vim nsys.sh
  ```
  
2. Write this in **nsys.sh**:
  ```sh
  nsys profile -t cuda,osrt,nvtx,cudnn,cublas -o test -w true --force-ovewrite true python3 predict.py
  ```
  
3. Run **nsys.sh**:
  ```sh
  source nsys.sh
  ```

Now the **test.nsys-rep** file is created in your current repository. Open this file using Nsight Systems API on your labtop (You can install the API following the instructions in https://developer.nvidia.com/nsight-systems. Unfortunately, you can't open the API on Jetson Nano.)  
  
<br/>
    
## Credits
- Minseo Kwon (@Minseo10)  
- Sooyoung Kwon (@Sooyoungk01)  
- Hyojin Kim (@hjyion)  

