# MRI-Super-Resolution

1. use smooth L1 to swap out L1 in loss fuction, it should make the training more stable by reducing the shake of loss decreasing

2. do NOT use fully connected layer in the end, but use conv2D in the end to make a FCN(fully convolutional netwok)

3. interpolation first to increase resolution little bit before leaving into U-Net(might NOT be useful since LR_MRI_image has same size as HR_MRI_image, LR_MRI_image just does NOT have high frequency information comepared with HR_MRI_image so tp speak)
        
4. @TODO now
        
        1.将小图拼接起来成为大图看下
        
        2.LR的SSIM似乎太低，调高一些再试一试
        
        
        3.7) might consider just using residual link in the shallow layers rather than using residual link in both shallow layers(e.g. first 30% layers) and deep layers(e.g. the deeper 70% layers), the reason is only residual link in shallow layers are actually passing the value actively according to [5]
        
        4.换一个更好的upsampling的方案（有其他函数可以用，应该有一种比pixelshuffler更牛逼）
        
        5.引入dropout，防止overfitting看下效果
        
        6.10)might also consider changing the order of connection, from "Conv --> BN --> ReLU"(normal connection) to "BN --> ReLU --> Conv"(so called full pre-activation)[13] The authors of ResNet[1] found out the performance increased if order of connection changed to full pre-activation in [13]. However, note the BN should always be placed before ReLU or other activation functions, due that "BN is used to produce activations function with the desired distribution"[14] so it has to be before activate function
        
        
        7.需要设计一个更好的loss来真实反映人对于超清的感受
        
20190720新任务
1. 重新生成LR图像，对2D网络图像只做层内模糊
2. 解决过拟合
3. 对更多细节部分增加ssim权重
4. pytorch上ssim值过大，检查或重写
