# MRI-Super-Resolution

1. use smooth L1 to swap out L1 in loss fuction, it should make the training more stable by reducing the shake of loss decreasing

2. do NOT use fully connected layer in the end, but use conv2D in the end to make a FCN(fully convolutional netwok)

3. interpolation first to increase resolution little bit before leaving into U-Net(might NOT be useful since LR_MRI_image has same size as HR_MRI_image, LR_MRI_image just does NOT have high frequency information comepared with HR_MRI_image so tp speak)
        
20190720新任务
1. 重新生成LR图像，对2D网络图像只做层内模糊。并且需要提高生成的LR的SSIM（比如保留频域的1/8而不是像现在只保留1/32）
2. 解决过拟合
3. 对更多细节部分增加ssim权重(如何用数量的方式表征"细节比较多"，比如概率？在loss里面需要重新设计一下对细节比较多的部分增加weight，类似分类任务的focal loss的方式)
4. pytorch上ssim值过大，检查或重写
5. 将小图拼接起来成为大图看下
6. 换一个更好的upsampling的方案（有其他函数可以用，应该有一种比pixelshuffler更牛逼）
7. 需要设计一个更好的loss来真实反映人对于超清的感受
8. 10)might also consider changing the order of connection, from "Conv --> BN --> ReLU"(normal connection) to "BN --> ReLU --> Conv"(so called full pre-activation)[13] The authors of ResNet[1] found out the performance increased if order of connection changed to full pre-activation in [13]. However, note the BN should always be placed before ReLU or other activation functions, due that "BN is used to produce activations function with the desired distribution"[14] so it has to be before activate function
