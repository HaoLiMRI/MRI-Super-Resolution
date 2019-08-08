# MRI-Super-Resolution

1. use smooth L1 to swap out L1 in loss fuction, it should make the training more stable by reducing the shake of loss decreasing

2. do NOT use fully connected layer in the end, but use conv2D in the end to make a FCN(fully convolutional netwok)

3. interpolation first to increase resolution little bit before leaving into U-Net(might NOT be useful since LR_MRI_image has same size as HR_MRI_image, LR_MRI_image just does NOT have high frequency information comepared with HR_MRI_image so tp speak)
        
## 最近新任务
1. ~~重新生成LR图像，对2D网络图像只做层内模糊。并且需要提高生成的LR的SSIM~~
        
        已重新生成2D 1/4，1/6，1/8 LR图像。
2. 解决过拟合
        
        现在已经在最后一层之后带dropout,可以调大dropout prob
        
        L2 regularization。但有论文说不应该只是weight decay实现L2 regularization，而应该改这个方式。https://arxiv.org/pdf/1711.05101.pdf
	
		early stop。监测test data的loss的趋势，每次training之后都看一下test data的loss。之后需要在大概test data的loss不降却上升时候stop。
        
        L1 regularization。好像需要手动加，见 https://zhuanlan.zhihu.com/p/69339955
        在PyTorch中还没有直接设置L1 范数的方法，可以在训练时Loss做BP之前（也就是.backward()之前）手动为Loss 加上L1范数：
		# 为Loss添加L1正则化项
		L1_reg = 0
		for param in net.parameters():
			L1_reg += torch.sum(torch.abs(param))
		loss += 0.001 * L1_reg  # lambda=0.001           
        用上节的代码试了一下，使用L1正则化项时如果指定和使用L2 正则化项时相同的λ=0.01 会发生under-fitting，似乎如果要用L1 正则化的话要把其系数设置的小一点，所以这里用了0.001。
        
        多加几个dropout试下
	
		减少网络的深度

		减小batch size也可以一定程度防止overfitting。
		
batch size和学习率如何影响网络的性能
链接：https://www.zhihu.com/question/32673260/answer/675161450
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

可以参考这篇文章：龙鹏-言有三：【AI不惑境】学习率和batchsize如何影响模型的性能？​zhuanlan.zhihu.com目前深度学习模型多采用批量随机梯度下降算法进行优化，随机梯度下降算法里n是批量大小(batchsize)，η是学习率(learning rate)。可知道除了梯度本身，这两个因子直接决定了模型的权重更新，从优化本身来看它们是影响模型性能收敛最重要的参数。学习率直接影响模型的收敛状态，batchsize则影响模型的泛化性能，两者又是分子分母的直接关系，相互也可影响，因此这一次来详述它们对模型性能的影响。模型性能对batchsize虽然没有学习率那么敏感，但是在进一步提升模型性能时，batchsize就会成为一个非常关键的参数。（关于学习率对模型性能的影响，可以参考上文）1 大的batchsize减少训练时间，提高稳定性这是肯定的，同样的epoch数目，大的batchsize需要的batch数目减少了，所以可以减少训练时间，目前已经有多篇公开论文在1小时内训练完ImageNet数据集。另一方面，大的batch size梯度的计算更加稳定，因为模型训练曲线会更加平滑。在微调的时候，大的batch size可能会取得更好的结果。2 大的batchsize导致模型泛化能力下降在一定范围内，增加batchsize有助于收敛的稳定性，但是随着batchsize的增加，模型的性能会下降，来自于文[5]。这是研究者们普遍观测到的规律，虽然可以通过一些技术缓解。这个导致性能下降的batch size在上图就是8000左右。那么这是为什么呢？研究[6]表明大的batchsize收敛到sharp minimum，而小的batchsize收敛到flat minimum，后者具有更好的泛化能力。两者的区别就在于变化的趋势，一个快一个慢，造成这个现象的主要原因是小的batchsize带来的噪声有助于逃离sharp minimum。Hoffer[7]等人的研究表明，大的batchsize性能下降是因为训练时间不够长，本质上并不少batchsize的问题，在同样的epochs下的参数更新变少了，因此需要更长的迭代次数。总之batchsize在变得很大(超过一个临界点)时，会降低模型的泛化能力。在这个临界点之下，模型的性能变换随batch size通常没有学习率敏感。3 学习率和batchsize的关系通常当我们增加batchsize为原来的N倍时，要保证经过同样的样本后更新的权重相等，按照线性缩放规则，学习率应该增加为原来的N倍[5]。但是如果要保证权重的方差不变，则学习率应该增加为原来的sqrt(N)倍[7]，目前这两种策略都被研究过，使用前者的明显居多。从两种常见的调整策略来看，学习率和batchsize都是同时增加的。学习率是一个非常敏感的因子，不可能太大，否则模型会不收敛。同样batchsize也会影响模型性能，那实际使用中都如何调整这两个参数呢？研究[8]表明，衰减学习率可以通过增加batchsize来实现类似的效果，这实际上从SGD的权重更新式子就可以看出来两者确实是等价的，文中通过充分的实验验证了这一点。研究[9]表明，对于一个固定的学习率，存在一个最优的batchsize能够最大化测试精度，这个batchsize和学习率以及训练集的大小正相关。对此实际上是有两个建议：如果增加了学习率，那么batch size最好也跟着增加，这样收敛更稳定。尽量使用大的学习率，因为很多研究都表明更大的学习率有利于提高泛化能力。如果真的要衰减，可以尝试其他办法，比如增加batch size，学习率对模型的收敛影响真的很大，慎重调整。

参考文献
[1] Smith L N. Cyclical learning rates for training neural networks[C]//2017 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2017: 464-472.

[2] Loshchilov I, Hutter F. Sgdr: Stochastic gradient descent with warm restarts[J]. arXiv preprint arXiv:1608.03983, 2016.

[3] Reddi S J, Kale S, Kumar S. On the convergence of adam and beyond[J]. 2018.

[4] Keskar N S, Socher R. Improving generalization performance by switching from adam to sgd[J]. arXiv preprint arXiv:1712.07628, 2017.

[5] Goyal P, Dollar P, Girshick R B, et al. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.[J]. arXiv: Computer Vision and Pattern Recognition, 2017.

[6] Keskar N S, Mudigere D, Nocedal J, et al. On large-batch training for deep learning: Generalization gap and sharp minima[J]. arXiv preprint arXiv:1609.04836, 2016.

[7] Hoffer E, Hubara I, Soudry D. Train longer, generalize better: closing the generalization gap in large batch training of neural networks[C]//Advances in Neural Information Processing Systems. 2017: 1731-1741.

[8] Smith S L, Kindermans P J, Ying C, et al. Don't decay the learning rate, increase the batch size[J]. arXiv preprint arXiv:1711.00489, 2017.

[9] Smith S L, Le Q V. A bayesian perspective on generalization and stochastic gradient descent[J]. arXiv preprint arXiv:1710.06451, 2017.
        	
		
3. 对更多细节部分增加ssim权重(如何用数量的方式表征"细节比较多"，比如概率？在loss里面需要重新设计一下对细节比较多的部分增加weight，类似分类任务的focal loss的方式)
        
        李昊老板建议gradient based SSIM。要看一下这个是什么，用下
        a) http://kresttechnology.com/krest-academic-projects/krest-mtech-projects/ECE/dspmt/[36].pdf
        b) https://ieeexplore.ieee.org/abstract/document/4107183
        基于边缘检测，计算强度变化速率，包含contrast和structure信息。均匀图片内gradient总和很小，对比度大的组织交界面会产生很大的梯度。
        目前实验中，网络为了匹配l和c产生的很强的噪声，噪点和周围像素之间也会产生梯度，使图片内梯度的总和和分布与原图不同。所以使用文章b中对gradient
        map求c和s的方法应该会进一步体现出这种差异。
        但是对于没有复杂结构的图片，整体梯度总和很小，梯度图直接的差异可能也很小，在batch中做平均还是会降低复杂图片产生的loss。
        不过可以根据图片的梯度总和判断图片内结构的复杂程度，在做平均的时候加上更高的权重。
        
4. ~~pytorch上ssim值过大，检查或重写(Check this repo: https://github.com/chisyliu/srMRI_SRGAN-2/blob/master/pytorch_ssim/__init__.py)~~
        
        已经换为pytorch_SSIM来构造SSIM
        
5. 将小图拼接起来成为大图看下.对于LR大图，SR大图还有HR大图都计算一下SSIM，看看是否SR大图与HR大图SSIM足够高而LR大图相对低

6. ~~换一个更好的upsampling的方案（有其他函数可以用，应该有一种比pixelshuffler更牛逼）~~
        
        各种upsampling技术见这 https://blog.csdn.net/g11d111/article/details/82855946
        另外注意，Meta-SR其实主要也是提出了一种基于Meta learning的upscale module/upsampling的方案
        新增transpose conv来做upsampling。具体见此https://blog.csdn.net/tsyccnh/article/details/87357447
        
7. 需要设计一个更好的loss来真实反映人对于超清的感受
   
   不同loss function的比较: 
   https://arxiv.org/abs/1511.08861
   https://www.sciencedirect.com/science/article/pii/S1047320319301336

8. 10)might also consider changing the order of connection, from "Conv --> BN --> ReLU"(normal connection) to "BN --> ReLU --> Conv"(so called full pre-activation)[13] The authors of ResNet[1] found out the performance increased if order of connection changed to full pre-activation in [13]. However, note the BN should always be placed before ReLU or other activation functions, due that "BN is used to produce activations function with the desired distribution"[14] so it has to be before activate function
9. ~~重新构建SSIM loss，首先用 "a x log(C) + b x log(L) + c x log(S)"方案构架,跑200epoches后看一看C,L,S哪一个很高不怎么降，之后再设置对应的weight更高来加快penalize这个对应的东西。看上去S好像性能提升的能力最差，所以应该加大相关weight，penalize这个部分loss效果~~

        已跑。性能不好.现在改用SSIM和1的L1 error function来做loss
        
10. ~~MS-SSIM跑一下看看性能~~
        
        已跑，性能非常烂
11. 研究一下另一个imgae content based SSIM(IW-SSIM)，用这个作为SSIM loss或者用SSIM和image content based SSIM一起作为SSIM loss。
        
        需要实现一个pytorch image content based SSIM。两个方案
        
                a. 找现成pytorch image content based SSIM代码，用对方代码直接作为我们需要的代码
                
                b. 或者自己写，在pytorch_SSIM上修改
12. ~~去掉所有的batch norm.~~

        搞定
        according to DDBPN paper, "Unlike the original DenseNets, we avoid dropout and batch norm, which are not suitable for SR, because they remove the range flexibility of the features [31]. Instead, we use 1 x 1 convolution layer as feature pooling and dimensional reduction [42, 12] before entering the projection unit.
        
        In paper which proposed EDSR; MDSR, 2017. Enhanced Deep Residual Networks for Single Image Super-Resolution, authors claim "Since batch normalization layers normalize the features, they get rid of range flexibility from networks by normalizing the features, it is better to remove them"
13. 换个optimization algorithm, e.g. SGD with momentum and weight decay.

        不同optimizer比较  https://zhuanlan.zhihu.com/p/62585696

        SGD 算法虽然简洁，但其在神经网络训练中的性能堪比高级二阶优化方法。尽管 SGD 每一次用小批量算出来的更新方向可能并非那么精确，但更新多了效果出乎意料地好。
        一般而言，SGD 各种变体可以分成两大类：1）自适应学习率机制，如 AdaGrad 和 Adam；2）加速机制，如 Polyak heavyball 和 Nesterov momentum 等。这两种方法都利用之前累积的梯度信息实现快速收敛，它们希望借鉴以往的更新方向。但是，要想实现神经网络性能提升，通常需要花销高昂的超参数调整。
        其实很多研究者都发现目前的最优化方法可能有些缺点，不论是 Adam 还是带动量的 SGD，它们都有难以解决的问题。例如我们目前最常用的 Adam，我们拿它做实验是没啥问题的，但要是想追求收敛性能，那么最好还是用 SGD+Momentum。但使用动量机制又会有新的问题，我们需要调整多个超参数以获得比较好的效果，不能像 Adam 给个默认的学习率 0.0001 就差不多了。
        在 ICLR 2018 的最佳论文 On the Convergence of Adam and Beyond 中，研究者明确指出了 Adam 收敛不好的原因。他们表明在利用历史梯度的移动均值情况下，模型只能根据短期梯度信息为每个参数设计学习率，因此也就导致了收敛性表现不太好。


14. ~~最近有一个新的optimization alg称之为Lookahead Optimizer据说非常牛逼，甚至不需要怎么精细的设置optimizer的参数就可以有比较好的效果~~
        
        DONE lookahead opt
        论文见这 https://arxiv.org/pdf/1907.08610v1.pdf
        资料 https://zhuanlan.zhihu.com/p/75184359
        Pytorch下Lookahead Optimizer实现 https://github.com/chisyliu/lookahead.pytorch
        Jianan看一下这个怎么实现的，确认是否可以移植过来用。如果真的如这篇paper作者讲的那样好，则有可能解决现在loss降不下去的问题


15. 论文"Structure, Frequency and Perceptual Refinement for MRI Super Resolution Reconstruction with UResNeXt"每一部分谁来负责更新已经分配完毕。20190729这一周开始Jianan Liu和Hao Li开始分别写作各自负责的部分，同时更新相对应reference。

        论文地址 https://www.overleaf.com/project/5d3cb9370f7d706eca0260cc
        
        
16. ~~设计新的network model-2D_RDN_Based_MRI_SR与2D_DDBPN_Based_MRI_SR。用RDN与DDBPN网络作为新的MRI SR的网络模型并完成python file~~

17. ~~增加gradient map based L1 loss~~
        
        根据这篇论文https://ieeexplore.ieee.org/abstract/document/4107183，计算gradient map。再对SR的gradient map和HR的gradient map求minimize L1 error function
        
        已加
        
18. ~~调通2D_RDN_Based_MRI_SRR，使其可以生成变大size或者不变size的SR。~~

        已跑

18. 调通2D_DDBPN_Based_MRI_SR，使其可以生成变大size或者不变size的SR。

18. 调通2D_RCAN_Based_MRI_SR，使其可以生成变大size或者不变size的SR。

19. 基于2D_RDN_Based_MRI_SR,2D_DDBPN_Based_MRI_SRh跟2D_RCAN_Based_MRI_SR，用一样的size的LR图像和HR图像训练，尝试各种loss等方案领其生成足够好SR图像。

20. 基于2D_RDN_Based_MRI_SR,2D_DDBPN_Based_MRI_SR还有2D_RCAN_Based_MRI_S，用小size的LR图像和大size的HR图像训练，尝试各种loss等方案领其生成足够好的大size的SR图像。之后做transfer learning，固定网络多数parameters只令最后一部分可变trainable(e.g. 最后的upscaling layer trainable)，用大size的HR图像跟更大size的SHR图像训练，让网络生成更大size的足够好的ESR(Enhanced Super Resolution)图像。

21. 基于论文Image Enhancement by Recurrently-trained Super-resolution Network (https://arxiv.org/abs/1907.11341) 提供的recurrently training 思路来生成比HR图像更牛逼的ESR(Enhanced Super Resolution)图像

22. MRI Segmentation. 好像主流应该就是用U-Net来做MRI segmentation (U-Net见 U-Net: Convolutional Networks for Biomedical Image Segmentation https://arxiv.org/abs/1505.04597). 根据jianan现在的理解，如果有标注好的训练数据，完全可以直接用RDN和DDBPN的网络结构直接做training来做segmentation。如何设计一个好的loss function可能是重点。



-----
## 每天抽一些时间读以下的论文，讨论：
- Progressive Perception-Oriented Network for Single Image Super-Resolution
>> 论文下载 https://arxiv.org/abs/1907.10399
>> Code下载 https://github.com/Zheng222/PPON
>> Comment or Question: 这篇赶快看看，他好像除了设计了一个GAN网络之外还用了SSIM和MS-SSIM作为loss，仔细看一下他怎么设计的loss（好像是对1和MS-SSIM取L1）。大致看了一下他的代码，可以看到他的里面有VGG feature extractor,GAN,residual block,sub-pixel conv upsampling(pixel shuffler)，然后GAN的generator用的是他所谓的PPON（progressively upsampling）

- Coupled-Projection Residual Network for MRI Super-Resolution
>> 论文下载 https://arxiv.org/pdf/1907.05598.pdf
>> Comment or Question: 这篇赶快看看，人家怎么插值得到的图像的SSIM才只有不到0.5，但之后啥网络生成的SR都有0.9以上（但说的应该是整张图的，感觉我们真的需要把整个图拼起来看一看并且算一下SSIM）

- Efficient and Accurate MRI Super-Resolution using a Generative Adversarial Network and 3D Multi-Level Densely Connected Network
>> 论文下载 https://arxiv.org/ftp/arxiv/papers/1803/1803.01417.pdf
>> Comment or Question: 这篇赶快看看，还是UCLA的那个中国人用GAN做的，然后好像generator是什么resnet和densenet揉到一块的。你看他最后那个figure2的细节感觉就和我们的生成的细节差别不太大，好像稍微好一些可是人家用的是GAN。我说细节差距不很大是因为你看也是像我们的一样明亮对比挺明显，但好像结构相似上更好一点。不过有可能是某些特定的细节部分图像效果比较好，他选择了（比如像我们training的13th）

- stanford
https://med.stanford.edu/bmrgroup/Publications/PublicationHighlights/super_resolution_oa_biomarkers.html
>> Comment or Question: 我觉得细节效果并没有我们现在的好

- ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
>> 论文下载 https://arxiv.org/pdf/1809.00219.pdf
>> Comment or Question: ESRGAN来做计算机视觉SR的original paper.这个文章提出了一种挺牛逼的想法。做一个PSNR导向的GAN模型和SR，再微调（怎么做？用什么数据微调？）得到一个GAN导向的GAN模型和SR，之后他们给了两种方案-1. 对两个网络的参数进行插值，得到一个新的模型的参数，用这个新的参数的模型来生成SR。2.对两个模型的SR进行插值，得到一个新SR。据他们说第一种方案好。
感觉我们可以参考这个思路，做两个网络一个PSNR+视觉导向，一个MRI细节（称之为fidelity）导向。但可能需要直接对两个生成SR进行插值生成一个新的SR。

- Compressed Sensing MRI Reconstruction using a Generative Adversarial Network with a Cyclic Loss
>> 论文下载 https://arxiv.org/pdf/1709.00753.pdf
>> Comment or Question: 用cycle GAN做SR MRI

- Translating and Segmenting Multimodal Medical Volumes with Cycle- and Shape-Consistency Generative Adversarial Network
>> 论文下载 https://arxiv.org/pdf/1802.09655.pdf
>> Comment or Question: 这篇论文好像非常牛，用cycle GAN做CT到MRI或MRI到CT的生成。

- Brain MRI super-resolution using 3D generative adversarial networks
>> 论文下载 https://openreview.net/pdf?id=rJevSbniM
>> Comment or Question: 

- Channel Splitting Network for Single MR Image Super-Resolution
>> 这篇论文里LR的SSIM数值与我们的比较接近
>> 论文下载：https://arxiv.org/abs/1810.06453

- Single MR Image Super-Resolution via Channel Splitting and Serial Fusion Network
>> 论文下载：https://arxiv.org/abs/1810.06453
>> 和上面同一个作者，思路类似

- Compressed Sensing MRI via a Multi-scale Dilated Residual Convolution Network
>> 论文下载 https://arxiv.org/abs/1906.05251

- MRI Super-Resolution with Ensemble Learning and Complementary Priors
>> 论文下载 https://arxiv.org/abs/1907.03063

- Learned Image Downscaling for Upscaling using Content Adaptive Resampler
>> 论文下载 https://arxiv.org/abs/1907.12904
>> Code下载 https://github.com/sunwj/CAR
>> Comment or Question: 这个东西有啥用？设计了一个网络从HR生成LR，再用EDSR生成SR

- Image Enhancement by Recurrently-trained Super-resolution Network
>> 论文下载 https://arxiv.org/abs/1907.11341
>> Comment or Question: 这个好像生成了一个比HR更牛逼更高分辨率的SR图像?为啥可以这么牛逼

- Hybrid Residual Attention Network for Single Image Super Resolution
>> 论文下载 https://arxiv.org/abs/1907.05514
>> Comment or Question: channel attention. 注意这篇论文作对比用的网络都是感觉比较弱的网络模型来做reference

- ECCV. Deep Residual Attention Network for Spectral Image Super-Resolution
>> 论文下载 http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Shi_Deep_Residual_Attention_Network_for_Spectral_Image_Super-Resolution_ECCVW_2018_paper.pdf
>> Comment or Question: channel attention. 注意这篇论文作对比用的网络都是感觉比较弱的网络模型来做reference

- SELF SUPER-RESOLUTION FOR MAGNETIC RESONANCE IMAGES
>> 论文下载：https://link.springer.com/content/pdf/10.1007%2F978-3-319-46726-9_64.pdf
>> 论文下载：https://arxiv.org/pdf/1802.09431.pdf
>> Comment or Question: 








-----
## 长期计划：
1. 2D SR CNN各种网络
	
	参考这个ranking
	 
	https://paperswithcode.com/sota/image-super-resolution-on-set5-4x-upscaling
	 
        阅读这几篇论文

        2017. Enhanced Deep Residual Networks for Single Image Super-Resolution
                
                EDSR.MDSR论文:https://arxiv.org/abs/1707.02921
                
                代码在这:https://github.com/chisyliu/EDSR-PyTorch
        
        2018. Residual Dense Network for Image Super-Resolution
                
                RDN论文:https://arxiv.org/abs/1802.08797
                
                代码在这:https://github.com/chisyliu/RDN
                
                Comment:
                        去掉了所有的batch norm;
                        用了一些经常都可以看到的所谓的global residual link;
                        把dense block后的output与dense block的input又做了一下summation于是就结合了resnet block变为了residual dense block;
                        还把每一个Residual dense block的output最后都concatenate到一起。
                
         2018. Deep Back-Projection Networks For Super-Resolution 
         
                DDBPN论文:https://arxiv.org/abs/1803.02735
                
                代码在这:https://github.com/chisyliu/DBPN-Pytorch
                
                Comment:
                        这篇paper不用sub pixel conv(pixel shuffle)来实现upsampling，而是用transpose conv实现upsampling。
                        关于transpose conv具体内容见https://blog.csdn.net/tsyccnh/article/details/87357447
                        这篇paper没有像EDSR, RDN，以及我们现在用的based on U-ResNeXt的模型一样在后半部分用single upsampling module，也不是像某篇韩国人paper中那样用predefined upsampling在进入网络之前就interpolate生成一个upsampling的图像，也没有做所谓progressive upsampling来逐渐step by step进行upsampling，而是设计了所谓iterative up and downsampling来upsampling downsampling upsampling downsampling这样来做
          
         2019. Progressive Perception-Oriented Network for Single Image Super-Resolution
         
                PPON论文:https://arxiv.org/abs/1907.10399
                
                代码在这:https://github.com/chisyliu/PPON
		
	 	2018. Image Super-Resolution Using Very Deep Residual Channel Attention Networks
		
			RCAN论文:https://arxiv.org/pdf/1807.02758v2.pdf
		
			代码在这:https://github.com/chisyliu/RCAN
		
		2019. Feedback Network for Image Super-Resolution
		
			SRFBN论文:https://arxiv.org/abs/1903.09814
		
			代码在这:https://github.com/chisyliu/SRFBN_CVPR19
			
			
		2019. Gated Multiple Feedback Network for Image Super-Resolution
		
			GMFN论文:https://arxiv.org/pdf/1907.04253v2.pdf
			
			代码在这:https://github.com/chisyliu/GMFN
		
		
                
   这几个网络都可以直接用来做MRI SR重建(特别是第二个，实现起来应该比较简单)，不过应该是对LR和HR,SR的size不一样的场景的SR重建
   
2. 2D 2DSRGAN(using U-ResNeXt) MRI SR

		original SRGAN论文 https://arxiv.org/abs/1609.04802

		SRGAN和SRWGAN的代码在这 https://github.com/chisyliu/SRGAN_WGAN-PyTorch
	 	
		SRGAN的代码在这 https://github.com/chisyliu/PyTorch-SRGAN
		
		https://github.com/chisyliu/srMRI_SRGAN-2
	 
3. 2D 2DESRGAN(using other feature extractor based generator, e.g. RDN, DDBPN, etc, to replace the generator proposed in original paper: "Residual-in-Residual Dense Block (RRDB) without batch normalization layers".) MRI SR

		根据ESRGAN作者描写，
		We improve the SRGAN from three aspects:
		
		a. adopt a deeper model using Residual-in-Residual Dense Block (RRDB) without batch normalization layers.这个应该是指在generator内的修改
		b. employ Relativistic average GAN instead of the vanilla GAN.
		这个应该是指在discriminator内的修改，使用了一种relativistic discriminator which uses relative probability than absolute probability。具体见这个文章 Relativistic GAN:https://ajolicoeur.wordpress.com/relativisticgan/.  https://arxiv.org/pdf/1807.00734.pdf
		Relativistic discriminator代码见 https://github.com/chisyliu/RelativisticGAN
		c. improve the perceptual loss by using the features before activation.
		d. 这个文章提出了一种挺牛逼的想法。做一个PSNR导向的GAN模型和SR，再微调（怎么做？用什么数据微调？）得到一个GAN导向的GAN模型和SR，之后他们给了两种方案-1. 对两个网络的参数进行插值，得到一个新的模型的参数，用这个新的参数的模型来生成SR。2.对两个模型的SR进行插值，得到一个新SR。据他们说第一种方案好。 感觉我们可以参考这个思路，做两个网络一个PSNR+视觉导向，一个MRI细节（称之为fidelity）导向。但可能需要直接对两个生成SR进行插值生成一个新的SR。
		
		In contrast to SRGAN, which claimed that deeper models are increasingly difficult to train, our deeper ESRGAN model shows its superior performance with easy training.
        
		original ESRGAN论文 https://arxiv.org/abs/1809.00219
	
        original ESRGAN code is: https://github.com/chisyliu/ESRGAN
	
		ESRGAN作者的另一版code https://github.com/xinntao/BasicSR
		
		我们考虑设计的网络要做的任务应该包括，将ESRGAN, WGAN-GP, 以及DAGAN for MRI SR的网络的优势都集合在一块
	
	
3. 3D U-ResNeXt MRI SR
4. 3D 3DSRGAN(using U-ResNeXt) MRI SR
5. Enhanced Self super resolution: Down-sized LR images are generated from HR images, and used to train the neural network. Use the trained neural network to process HR images and produce higher resolution images   
        
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
        
7. MRI Segmentation. 好像主流应该就是用U-Net来做MRI segmentation (U-Net见 U-Net: Convolutional Networks for Biomedical Image Segmentation https://arxiv.org/abs/1505.04597).
根据jianan现在的理解，如果有标注好的训练数据，完全可以直接用RDN和DDBPN的网络结构直接做training来做segmentation。如何设计一个好的loss function可能是重点。

        如果没有标注好的训练数据，可以考虑用open data set: LGG MRI Segmentation Dataset  https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

8. Super resolution + segmentation，文章很少，几乎没有MRI相关的
- Highly Accurate Facial Nerve Segmentation Refinement From CBCT/CT Imaging Using a Super-Resolution Classification Approach
  下载地址：https://ieeexplore.ieee.org/document/7911203
- Utility of Deep Learning Super-Resolution in the Context of Osteoarthritis MRI Biomarkers，stanford今年新的文章，里面有两个网络，一个SR，一个segmentation。
   下载地址：https://onlinelibrary.wiley.com/doi/pdf/10.1002/jmri.26872
   
9. 多channel训练，然后做平均，也需可以修正对分布的估计。
   由Channel Splitting Network for Single MR Image Super-Resolution想到的
   下载地址：https://arxiv.org/pdf/1810.06453.pdf
   
-----
 
 
 
 
 
 **Note:**
 1. 我们现在设计基于U-ResNeXt的网络是处理LR与HR,SR都有相同的size的case。而EDSR, RDN, Meta-SR都应该是handle输入LR与HR,SR的size有r倍差距的case
 2. 总结来看，所有的SR图像重构网络基本都是两个部分组成。第一部分是feature learing module,可以基于resnet, densenet, U-Net的前半部分，等等.第二部分是upscale module,可以基于EDSR，RDN里面的sup pixel convolution(sup pixel convolution就是我们正在用的U-Net的后半部分的pixelshuffer), DDBPN里用的transpose conv，Meta-SR里面的Meta upscale module等等。
-----
        
   


**写作思路**
只针对(2D MRI SR)&&(Non-Self super resolution)，都有如下写作内容可以好多paper
1. 基于现有的Gated U-ResNeXt，加上各种loss，实现的是LR和HR，SR的size相同时候的MRI SR重构
2. 基于SRGAN，用Gated U-ResNeXt，加上各种loss，实现的是LR和HR，SR的size相同时候的MRI SR重构
3. 基于RDN，加上各种loss，实现的是LR和HR，SR的size不相同并且resize factor固定不任意的MRI SR重构
4. 基于RDN based Meta-SR，加上各种loss，实现的是LR和HR，SR的size不相同并且resize factor任意的MRI SR重构
5. 基于SRGAN，能否用RDN based Meta-SR来做generator或别的方案加入RDN based Meta-SR，加上各种loss，实现的是LR和HR，SR的size不相同并且resize factor任意的MRI SR重构
-----





**写作常见英文**
1. SCI写作常用句式总结一(Introduction篇)

	https://zhuanlan.zhihu.com/p/74664090
-----





**需要花时间阅读并且讨论总结的paper**
1. 非常重要，非常好总结 https://medium.com/beyondminds/an-introduction-to-super-resolution-using-deep-learning-f60aff9a499d
1. 2019. An overview of deep learning in medical imaging focusing on MRI https://www.sciencedirect.com/science/article/pii/S0939388918301181
2. Deep Learning for Image Super-resolution: A Survey https://arxiv.org/pdf/1902.06068.pdf
2. Deep Learning for Single Image Super-Resolution: A Brief Review https://arxiv.org/pdf/1808.03344.pdf
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
 
