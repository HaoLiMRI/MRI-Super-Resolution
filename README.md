# MRI-Super-Resolution

1. use smooth L1 to swap out L1 in loss fuction, it should make the training more stable by reducing the shake of loss decreasing

2. do NOT use fully connected layer in the end, but use conv2D in the end to make a FCN(fully convolutional netwok)

3. interpolation first to increase resolution little bit before leaving into U-Net(might NOT be useful since LR_MRI_image has same size as HR_MRI_image, LR_MRI_image just does NOT have high frequency information comepared with HR_MRI_image so tp speak)
        
## 20190722新任务
1. ~~重新生成LR图像，对2D网络图像只做层内模糊。并且需要提高生成的LR的SSIM~~
        
        已重新生成2D 1/4，1/6，1/8 LR图像。
2. 解决过拟合
        
        现在已经带dropout,但逻辑上dropout不应该对的。之后还要再研究结果
3. 对更多细节部分增加ssim权重(如何用数量的方式表征"细节比较多"，比如概率？在loss里面需要重新设计一下对细节比较多的部分增加weight，类似分类任务的focal loss的方式)
4. ~~pytorch上ssim值过大，检查或重写(Check this repo: https://github.com/chisyliu/srMRI_SRGAN-2/blob/master/pytorch_ssim/__init__.py)~~
        
        已经换为pytorch_SSIM来构造SSIM
5. 将小图拼接起来成为大图看下
6. 换一个更好的upsampling的方案（有其他函数可以用，应该有一种比pixelshuffler更牛逼）
        
        各种upsampling技术见这 https://blog.csdn.net/g11d111/article/details/82855946
        
        另外注意，Meta-SR其实主要也是提出了一种基于Meta learning的upscale module/upsampling的方案
        
7. 需要设计一个更好的loss来真实反映人对于超清的感受
8. 10)might also consider changing the order of connection, from "Conv --> BN --> ReLU"(normal connection) to "BN --> ReLU --> Conv"(so called full pre-activation)[13] The authors of ResNet[1] found out the performance increased if order of connection changed to full pre-activation in [13]. However, note the BN should always be placed before ReLU or other activation functions, due that "BN is used to produce activations function with the desired distribution"[14] so it has to be before activate function
9. 重新构建SSIM loss，首先用 "a x log(C) + b x log(L) + c x log(S)"方案构架,跑200epoches后看一看C,L,S哪一个很高不怎么降，之后再设置对应的weight更高来加快penalize这个对应的东西。
10. MS-SSIM跑一下看看性能
10. 研究一下另一个imgae content based SSIM(IW-SSIM)，用这个作为SSIM loss或者用SSIM和image content based SSIM一起作为SSIM loss。
        
        需要实现一个pytorch image content based SSIM。两个方案
        
                a. 找现成pytorch image content based SSIM代码，用对方代码直接作为我们需要的代码
                
                b. 或者自己写，在pytorch_SSIM上修改


-----
## 长期计划：
1. 2D RDN MRI SR和2D DDBPN MRI SR
        阅读这几篇论文

        2017. Enhanced Deep Residual Networks for Single Image Super-Resolution
                
                EDSR.MDSR论文:https://arxiv.org/abs/1707.02921
                
                代码在这:https://github.com/chisyliu/EDSR-PyTorch
        
        2018. Residual Dense Network for Image Super-Resolution
                
                RDN论文:https://arxiv.org/abs/1802.08797
                
                代码在这:https://github.com/chisyliu/RDN
                
                Comment:去掉了所有的batch norm,用了一些经常都可以看到的所谓的global residual link,把dense block后的output与dense block的input又做了一下summation于是就结合了resnet block变为了residual dense block，还把每一个Residual dense block的output最后都concatenate到一起。
                
         2018. Deep Back-Projection Networks For Super-Resolution 
         
                DDBPN论文:https://arxiv.org/abs/1803.02735
                
                代码在这:https://github.com/chisyliu/DBPN-Pytorch
                
                Comment:不用sub pixel conv(pixel shuffle)来实现upsampling，而是用transpose conv实现upsampling。关于trans
                pose conv具体内容见https://blog.csdn.net/tsyccnh/article/details/87357447
                
                
   这几个网络都可以直接用来做MRI SR重建(特别是第二个，实现起来应该比较简单)，不过应该是对LR和HR,SR的size不一样的场景的SR重建
   
2. 2D 2DSRGAN(using U-ResNeXt) MRI SR
3. 3D U-ResNeXt MRI SR
4. 3D 3DSRGAN(using U-ResNeXt) MRI SR
5. Self super resolution: Down-sized LR images are generated from HR images, and used to train the neural network. Use the trained neural network to process HR images and produce higher resolution images   
        
6. 阅读这篇论文2019. Meta-SR: Magnification-arbitrary network for Super resolution with variable scale factor(https://arxiv.org/abs/1903.00875 )。

        该文用meta learning的方式自适应的学习了一个weight prediction network的weights，用这个weight prediction network去预测upscale时所用的filter的weights。传统的fixed resize factor方案（e.g. EDSR, RDN, etc）都是用的sup pixel convolution做upscale module, sup pixel convolution filter的weights是从LR and HR training data放在EDSR or RDN + sup pixel convolution网络直接学来的。
        sup pixel convolution论文
          
          2016.Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network 下载（https://arxiv.org/pdf/1609.05158.pdf）
       
       
        相关资料 
        https://www.chainnews.com/articles/367464091791.htm
        https://blog.csdn.net/m0_37615398/article/details/88382556
        https://blog.csdn.net/m0_38129460/article/details/88596262 
   看一下怎么扩展到MRI SR Reconstruction。感觉可以直接拿过来用到MRI SR。Meta-SR用在我们的MRI SR话，具体的feature learning module可以考虑用各种feature extractor，比如ResNet，Gated-ResNeXt，EDSR, RDN。
   
        Meta-SR一作pytorch复现在这
                https://github.com/chisyliu/srMRI_Meta-SR-Pytorch
                其中复现了多种SR Reconstruction网络模型，包括
                a. EDSR (2017. Enhanced Deep Residual Networks for Single Image Super-Resolution https://arxiv.org/abs/1707.02921), 
                        复现在这https://github.com/chisyliu/EDSR-PyTorch
                b. MDSR (2017. Enhanced Deep Residual Networks for Single Image Super-Resolution https://arxiv.org/abs/1707.02921), 
                c. DDBPN (2018. Deep Back-Projection Networks For Super-Resolution https://arxiv.org/abs/1803.02735),
                d. RDN (2018. Residual Dense Network for Image Super-Resolution https://arxiv.org/abs/1802.08797),
                        相关资料 https://blog.csdn.net/qq_14845119/article/details/81459859                        
                e. RCAN (2018. Image Super-Resolution Using Very Deep Residual Channel Attention Networks https://arxiv.org/abs/1807.02758)                 
                并用这些模型分别作为Meta-SR的feature learning module来设计Meta-SR

   What is meta-learning?
        Meta-Learning: Learning to Learn Fast
        https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html
-----
 
 
 
 
 
 **Note:**
 1. 我们现在设计基于U-ResNeXt的网络是处理LR与HR,SR都有相同的size的case。而EDSR, RDN, Meta-SR都应该是handle输入LR与HR,SR的size有r倍差距的case
 2. 总结来看，所有的SR图像重构网络基本都是两个部分组成。第一部分是feature learing module,可以基于resnet, densenet, U-Net的前半部分，等等.第二部分是upscale module,可以基于EDSR，RDN里面的sup pixel convolution(sup pixel convolution就是我们正在用的U-Net的后半部分的pixelshuffer),Meta-SR里面的Meta upscale module等等。
-----
        
   


**写作思路**
只针对(2D MRI SR)&&(Non-Self super resolution)，都有如下写作内容可以好多paper
1. 基于现有的Gated U-ResNeXt，加上各种loss，实现的是LR和HR，SR的size相同时候的MRI SR重构
2. 基于SRGAN，用Gated U-ResNeXt，加上各种loss，实现的是LR和HR，SR的size相同时候的MRI SR重构
3. 基于RDN，加上各种loss，实现的是LR和HR，SR的size不相同并且resize factor固定不任意的MRI SR重构
4. 基于RDN based Meta-SR，加上各种loss，实现的是LR和HR，SR的size不相同并且resize factor任意的MRI SR重构
5. 基于SRGAN，能否用RDN based Meta-SR来做generator或别的方案加入RDN based Meta-SR，加上各种loss，实现的是LR和HR，SR的size不相同并且resize factor任意的MRI SR重构
-----




**需要花时间阅读并且讨论总结的paper**
1. 非常重要，非常好总结 https://medium.com/beyondminds/an-introduction-to-super-resolution-using-deep-learning-f60aff9a499d
2. Deep Learning for Single Image Super-Resolution: Overview https://arxiv.org/pdf/1808.03344.pdf
3. A Deep Journey into Super-resolution: A survey https://arxiv.org/pdf/1904.07523.pdf
4. Information Content Weighted Structural Similarity Index (IW-SSIM) for Image Quality Assessment: http://www.ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf
5. SSIM: http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
6. MS-SSIM: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
7. 多种方法对比：https://blog.csdn.net/qq_35860352/article/details/84037501
                https://blog.csdn.net/qq_23304241/article/details/80953613
8. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5527267/pdf/JMI-004-035501.pdf
9. 这篇文章https://zhuanlan.zhihu.com/p/39930043  总结了2018年SR比赛的一些算法，还提到了SRGAN比其他方法重建得到的图像要清晰，但在PSNR和SSIM上都要比其他方法甚至是bicubic上采用得到都要低很多。主要原因SRGAN使用了style transfer里用到的感知损失（当然也用非GAN方法使用感知损失的，例如EnhanceNet[8]），而感知损失重建的图像在人类的认知视觉上更舒服，但细节恢复上确实会和原图相差很多。所以在该文author理解，即便是超分辨率重建，依然可以将其分为两个方向。第一个方向力求恢复出真实可靠的细节部分，应用场景例如医学影像上的超分辨率重建，低分辨率摄像头人脸或者外形的恢复等对细节要求苛刻的场景。另一个则追求整体视觉效果，细节部位要求不高。例如低分辨率视频电视的恢复、相机模糊图像的恢复等。

-----




**interesting paper我们可以看一看，也许有帮助可以给我们参考或者可以用**
1. Lightweight Image Super-Resolution with Adaptive Weighted Learning Network https://arxiv.org/abs/1904.02358
        
        我们有Lightweight需求吗？
        
2. Efficient Deep Neural Network for Photo-realistic Image Super-Resolution https://arxiv.org/abs/1903.02240

3. Deep Learning for Multiple-Image Super-Resolution https://arxiv.org/abs/1903.00440
        
        不确定multiple-image SR具体是什么意思，需要看并且确认我们有需求吗？
        
4. Single MR Image Super-Resolution via Channel Splitting and Serial Fusion Network https://arxiv.org/abs/1901.06484

        可以看看别人怎么做MRI SR，可以启发一下我们的task不知道有没有可以对我们有意义的部分
-----



## A collection of high-impact and state-of-the-art SR method
https://github.com/chisyliu/Single-Image-Super-Resolution
 
