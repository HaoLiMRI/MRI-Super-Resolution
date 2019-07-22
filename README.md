# MRI-Super-Resolution

1. use smooth L1 to swap out L1 in loss fuction, it should make the training more stable by reducing the shake of loss decreasing

2. do NOT use fully connected layer in the end, but use conv2D in the end to make a FCN(fully convolutional netwok)

3. interpolation first to increase resolution little bit before leaving into U-Net(might NOT be useful since LR_MRI_image has same size as HR_MRI_image, LR_MRI_image just does NOT have high frequency information comepared with HR_MRI_image so tp speak)
        
## 20190722新任务
1. 重新生成LR图像，对2D网络图像只做层内模糊。并且需要提高生成的LR的SSIM
        
        已重新生成2D 1/4，1/6，1/8 LR图像。
2. 解决过拟合
        
        现在已经带dropout,但逻辑上dropout不应该对的。之后还要再研究结果
3. 对更多细节部分增加ssim权重(如何用数量的方式表征"细节比较多"，比如概率？在loss里面需要重新设计一下对细节比较多的部分增加weight，类似分类任务的focal loss的方式)
4. pytorch上ssim值过大，检查或重写(Check this repo: https://github.com/chisyliu/srMRI_SRGAN-2/blob/master/pytorch_ssim/__init__.py)
        
        已经换为pytorch_SSIM来构造SSIM
5. 将小图拼接起来成为大图看下
6. 换一个更好的upsampling的方案（有其他函数可以用，应该有一种比pixelshuffler更牛逼）
7. 需要设计一个更好的loss来真实反映人对于超清的感受
8. 10)might also consider changing the order of connection, from "Conv --> BN --> ReLU"(normal connection) to "BN --> ReLU --> Conv"(so called full pre-activation)[13] The authors of ResNet[1] found out the performance increased if order of connection changed to full pre-activation in [13]. However, note the BN should always be placed before ReLU or other activation functions, due that "BN is used to produce activations function with the desired distribution"[14] so it has to be before activate function
9. 研究一下另一个imgae content based SSIM，用这个作为SSIM loss或者用SSIM和image content based SSIM一起作为SSIM loss。
        
        需要实现一个pytorch image content based SSIM。两个方案
        
                a. 找现成pytorch image content based SSIM代码，用对方代码直接作为我们需要的代码
                
                b. 或者自己写，在pytorch_SSIM上修改



## 长期计划：
1. 2D 2DSRGAN(using U-ResNeXt)
2. 3D U-ResNeXt
3. 3D 3DSRGAN(using U-ResNeXt)
4. Self super resolution: Down-sized LR images are generated from HR images, and used to train the neural network. Use the trained neural network to process HR images and produce higher resolution images
5. 阅读这篇论文2019. Magnification-arbitrary network: Super resolution with variable scale factor(https://arxiv.org/abs/1903.00875 )。
        
        相关资料 
        https://www.chainnews.com/articles/367464091791.htm
        https://blog.csdn.net/m0_37615398/article/details/88382556
        https://blog.csdn.net/m0_38129460/article/details/88596262 
   看一下怎么扩展到MRI SR Reconstruction。感觉可以直接拿过来用到MRI SR。
        
        两个方案
                a. 要盯着代码什么时间放出来，在对方代码上面直接加我们的东西
                        (pytorch复现在这https://github.com/chisyliu/srMRI_Meta-SR-Pytorch)
                        相关模型EDSR.复现https://github.com/chisyliu/EDSR-PyTorch
                b. 或者自己写，jianan大概看了一下论文，感觉自己写应该没问题






## A collection of high-impact and state-of-the-art SR method
https://github.com/chisyliu/Single-Image-Super-Resolution
 
