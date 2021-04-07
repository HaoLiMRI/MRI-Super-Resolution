# MRI-Super-Resolution

3月1日开始的任务：TTSR with SSIM Loss, 4x TTSR, wavelet TTSR 2x and 4x.

有时间的话跑4，5，6。


## 至今方案的总结
主要包括以下几种方案：
1. loss function上设计更合理的loss项实现MRI SR， 例如SSIM loss, graident loss
2. 使用各种即插即用的模块，比如即插即用的conv模块(e.g. deformable conv)，activation function模块等
3. 网络结构上设计各种特征融合的构架，比如gradient map dual branch, k spacce dual branch, wavelet dual branch网络模型
4. 网络结构上用wavelet变换抓取高频特征，结合HR reference信息进行MRI SR恢复，例如我们自己的HR reference MRI SR网络模型
5. 网络结构上加入各种attention方案，比如加入CBAM的channel attention和spatial attention, kernel attention，和self-attention
6. 网络结构上仍然在每一层的H by W的MRI数据上结合Transformer抓取信息，结合HR reference信息进行MRI SR恢复，例如TTSR MRI SR网络模型
7. 试图在层间信息上进行处理或者相关性的抓取，例如直接用正常的3x3 conv with 3 channel的RCAN处理层间降采样（比如HR MRI图是128 x 128 x 6, LR MRI是在层间降采得出的128 x 128 x 3）的channel = 3的3D数据的MRI SR（又或者直接用正常的3x3 conv with 5 channel的RCAN处理层间降采样的channel = 5的3D数据的MRI SR）；又或者用Transformer或者其他网络（LSTM?）在层间纵向切面抓取特征的相关性来结合普通的层内抓取特征的比如RCAN来实现3D MRI SR


## 2021年的新idea和任务

### 第一篇和我们自己的HR reference网络至少到现在来看近期还要做的是
1. ~~给RCAN dual domain网络重新加一个最外面的long residual link，用zero padding放大LR输入让它跟HR统一尺寸。（如果效果好修改论文不光是网络结构部分要改，还有fig1。描述LR SR的理论部分2.1.）~~
2. ~~跑一下progressive和post upsampling的4x和8x放大~~
3. 跑一下deformable conv
4. 多跑12个左右的epoch，每个epoch都跑完一次完整的learning_rate_start至learning_rate_finish，然后把每个epoch跑完的model point存一下，最后求和取平均。有可能可以获得更好的效果。详见：https://mp.weixin.qq.com/s?__biz=MzIwMTE1NjQxMQ==&mid=2247551932&idx=2&sn=855a70ed0522a3abe571f4939a7511b5&chksm=96f079e8a187f0fecf0d2f18e978d2163568e8f2962577a7fe8f684d4f3c7c2bd0a1ad289df2&scene=132#wechat_redirect  这里所谓的“model point存一下，最后求和取平均”应该是指的将每个epoch跑完之后的model的参数求和平均。这种思路其实与ESRGAN论文(2018.Esrgan: Enhanced super-resolution generative adversarial networks,)中提到的network interpolation技术有差不多的概念（ESRGAN中提到的network interpolation：对GAN分别用PSNR导向的loss训练出一个可以恢复出相对而言较准确但平滑的有些过分的SR图像的网络参数A，和用对抗导向的loss训练出一个可以恢复出相对而言不准确但看起来很像真实图像不过分平滑的SR图像的网络参数B，再用网络参数A乘以p加上(1-p)乘以网络参数B求新的插值网络参数）。
5. 集成ensemble approach: 使用不同随机种子的学习RCAN MRI SR网络F1，…F10 —— 尽管具有非常相似的测试性能 —— 被观察到与非常不同的函数相关联。实际上，使用一种著名的技术叫做集成(ensemble)，只需对这些独立训练的网络的输出进行无加权的平均，就可以在许多深度学习应用中获得测试时性能的巨大提升。(参见下面的图1。)这意味着单个函数F1，…F10必须是不同的。见：https://mp.weixin.qq.com/s/YyLTd8B7M4f3hBTybrnUSQ
6. 自蒸馏 ：
方案一：通过对RCAN MRI SR的单个模型执行知识蒸馏（一个训练好的RCAN MRI SR模型做teacher，一个待训练的RCAN MRI SR模型做student，求俩者distillation loss。该loss可以先从L1 loss或者SSIM loss试验），测试的准确性也可以得到提高。
方案二：通过对RCAN MRI SR的已经做了ensemble的模型执行知识蒸馏（10个训练好的RCAN MRI SR模型已经用ensemble生成了SR，该loss可以先从L1 loss或者SSIM loss试验），测试的准确性也可以得到更多提高。
(见：https://mp.weixin.qq.com/s/YyLTd8B7M4f3hBTybrnUSQ 图2。)
7. ~~同样的RCAN模型再做de-motion artifact，把这个结果写进第一篇论文中~~。已完成。
8. RCAN模型基础上加入已经实现了的各种模块做de-motion artifact，要是结果很好则再写篇motion artifact removing论文。
9. ~~SR和HR分别求gradient map，再相减得到一个gradient map差的矩阵，再把这个gradient map差的矩阵从(H * W)变为(1 * HW)，然后再过一个softmax，再变回H * W，然后把得到的矩阵当做pixel-wise L1 loss的weight来元素乘在L1 loss的pixel上~~ 。
已完成，跑一下。跑过，没有明显效果。
10. ~~SR和HR求SSIM map，再用1减这个SSIM map得到一个矩阵当做pixel-wise L1 loss的weight来元素乘在L1 loss的pixel上~~ 。
已完成，跑一下。跑过，没有明显效果。
11. 看懂PC（Phase Congruency）怎么算，把这个指标做loss项。
参考论文：2011.FSIM: A Feature Similarity Index for Image Quality Assessment
参考代码：https://github.com/sunxirui310/FSIM-FSIMc-matlab/blob/master/FSIM.m
12. ~~跑一下channel and spatial attention on upsampler, 俩种framework（CBAM与self-attention）和俩种mode（并联串联）各自跑一下。~~
13. ~~把分别实现的并联和串联的"普通Channel and Spatial Attention Block"与"基于non local self-attention Channel and Spatial Attention Block"实现方案替换原RCAN中的CALayer，得到多个全新的模型再实验。~~
14. 如果可能的话，在代码中加入non local self-attention的channel and spatial attention的heatmap实现可视化。
15. 完成基于He Kaiming的paper: 2019.Panoptic Feature Pyramid Networks内figure 3提出的为semantic segmentation任务提出的Panoptic FPN方案来实现U-Net，并重复基于这种新的结合了Panoptic FPN的U-Net框架的MRI SR dual domain network。仍然是可以用以上所有实现的"Channel and Spatial Attention Block"替换CA Lyaer从而组成新的RCSAB，然后多个RCSAB构成新的RG，每个RG去替换U-Net原始框架中的每一层。然后跑一下这种Panoptic FPN的U-Net构架下的上面相同的各种实验。
16. ~~consider using HR reference with self-attention in the end. 使用MRI HR reference的MRI SR，写一个新的wrapper去并联两个现有的网络（比如两个attention based RCAN并联），一个用于LR的2倍放大，另一个用于给HR reference的feature extraction（去掉upsampler），最后用一个self-attention的upsampler来把俩者fuse到一起生成MRI SR。这个方案的思路是用CNN去抓取LR图像和HR图像的局部特征的feature，然后用self-attention方案去找到这些局部feature在整个图上（全局上，更大的范围）的关系。模型的结构可以参考Paper: 2020.Attention-based Image Upsampling. https://arxiv.org/abs/2012.09904 。这个结合self-attention在最后的利用HR reference的方案可以使用的训练数据和TTSR MRI SR的数据一样~~ 。
已完成，跑一下。
现在这个任务还有的遗留问题包括：
a). option to use HR reference with self-attention in the end这个方案已经代码已经完成。但现在只支持RCAN的image_single_domain或者gradient_map_dual_domain在没有long_skip_connection_to_reconstruct_residual_part_only时的HR reference based network。不支持其他配置时的HR reference based network。注意：现在gradient_map_dual_domain时用HR reference based network还有问题，会out of memory。
b). 另外，现在option to use HR reference with self-attention in the end这个方案如果在网络用self-attention，则会out of memory。
c). option to add long skip connection outside the entire network model to only reconstruct the residual part of HR MRI image这个选项现阶段仅支持非HR Reference based的网络结构，还不支持HR Reference based网络结构。

17.受paper: 2019.Local Relation Networks for Image Recognition中figure2的启发，那个图它引入了一个什么geometry prior，然后说要对each spatial position来做self-attention。我们可以像它这样，但不对每一个spatial position来做，而是在16)中的方案using HR reference with self-attention in the end那样最后做slef-attention的部分引入一个比如LR的图的Gradient map像它这个geometry prior一样加到self-attention里面。	

18 ~~Inverse-wavelet-only RCAN：参考A Wavelet-Based Asymmetric Convolution Network for Single Image Super-Resolution。基于当前的RCAN网络，从LR图像生成HR图像的小波分量，用IDWT代替上采样。代码中把head里的conv设置成padding=2，tail里的conv输出channel数设置为4，用IDWT代替上采样。~~ 已完成，跑的效果相对RCAN MRI SR没有改进。

19. 还是我们自己的HR reference网络，也应该让HR reference经过小波变换，然后只保留高频部分进入网络帮助LR做SR。方案二：对于我们自己的HR reference网络可以考虑对HR reference数据做DWT（小波变换），只保留3个高频分量。然后对LR数据像“Inverse-wavelet-only RCAN”这个idea一样处理，直接用网络试图从LR数据生成HR图像的小波分量，然后把用网络对LR数据生成的3个高频小波分量和FR reference做DWT得到的3个HR reference的高频分量进行fusion。最后再把LR数据用网络试图直接生成的一个低频分量和fuse后的3个高频分量一起做IDWT（小波反变换）恢复2x SR图像。如果是4x则用progressive构架重复2次2x放大的该方案。

20. 我们自己的HR reference网络，也应该让HR reference经过小波变换，然后只保留高频部分进入网络帮助LR做SR。方案一：对于我们自己的HR reference网络可以考虑把LR复制3份，分别于HR reference的小波变换的3个高频分量各自过self-attention一起组成multi-head self-attention。

21. 设计一个通用的“外挂”型纵向切面抓取feature的Transformer。因为MRI数据其实应该为3D数据，所以在相邻的几层间应该有可用的信息来帮助提升分辨率。所以在任何一个现有的2D横切面MRI数据的SR网络（e.g. RCAN MRI SR, TTSR MRI SR, Wavelet TTSR MRI SR）基础上都可以再增加一个单独的branch，这个branch读进2D LR MRI图像在原始3D图像中对应的前后N层（e.g.前后5层）的数据，在纵向切面（2N层上）用Transformer抓取feature，之后将这些feature在upsampling前与主branch（e.g. 原始RCAN MRI SR, TTSR MRI SR, Wavelet TTSR MRI SR）得到的feature map来fuse到一起再过Upsampling生成MRI SR。

22. 直接对SR和HR求KL散度loss。我们在做object detection时候有时候不直接估计Bounding box，而是先估计一个feature map或者说一个heat map（就是一个n, 1, h, w的矩阵），然后从这个heatmap提取K个最大值作为估计Object的中心点。所以这种时候估计heatmap的loss function一般都是用BCE loss，表示每个pixel有多大概率是Object的中心。所以这里其实就是把heatmap作为一个概率性的分布表征，求取Ground truth对应的那个概率性分布表征n, 1, h, w的矩阵和估计的那个概率性分布表征n, 1, h, w的矩阵的相似性。BCE loss其实就是KL散度的一种特殊情况。那对于我们的MRI SR任务，其实也是在求取Ground truth对应的那个概率性分布表征n, 1, h, w的矩阵和估计的那个概率性分布表征n, 1, h, w的矩阵的相似性，所以我们是不是可以直接把比如KL散度这种表示俩种分布相似性的准则做loss?

23. 受论文2021.Channel Boosting Feature Ensemble for Radar-based Object Detection图1的结构启发，我们可以把“MRI LR图和相应的HR reference图”，或者“MRI LR图和前后N层的MRI LR数据（self-reference）”，或者“MRI LR图和相应的小波变换之后的分量”，一起concatenate后输入RCAN再过upsampler（或者分别输入RCAN再过upsampler之后再concatenate到一块），然后再输给Transformer（Positioning Encoding，Multi-head self attention，MLP/FFN）提取相关性并输出最终的MRI SR结果（或者RCAN部分只做下采样，upsampler都放在经过Transformer后再upsampling）。

24. 我们自己的HR reference网络现在最后用了self-attention，我们应该在过self-attention后再加一个很小的RCAB（因为2020.Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth这篇论文说没有skip connection和MLP，self-attention layers啥也不是，所以我们考虑再后面再加一个类似FFN/MLP的小CNN）。


### 第一篇中长期还要做的是
1. ~~完成基于U-Net框架的MRI SR dual domain branch，把dual domain branch用以上所有实现的"Channel and Spatial Attention Block"替换CA Lyaer从而组成新的RCSAB，然后多个RCSAB构成新的RG，每个RG去替换U-Net原始框架中的每一层。然后跑一下这种U-Net构架下的上面相同的各种实验。~~
2. 我们做实验，对比 基于完成的RCSAB based U-Net框架的MRI SR dual domain network,以及RCSAB based U-Net with Panoptic FPN框架的MRI SR dual domain network和RCSAB based RCAN的网络的性能。对应的，论文中改为写我们为这俩大类网络结构（一种是以RCAN为最优的性能代表的channel一直不变的模型，例如EDSR,DDBPN,RCAN。另外一种是以U-Net为代表的channel先逐步放大再逐步缩小）for MRI SR做了比较 上升到这俩大类的网络构架哪个更好。因为这俩类网络结构一个是完全不缩小size，然后channel数量也不变 另一个是size先小后大，然后channel数量是逐步放大再逐步缩小。可以说是完全不同的俩类结构 我们这样对比完善实验 可以说是为MRI SR任务探索了俩种主流模型构架的方案哪个更靠谱。
3. k space, wavelet secondary branch多个分量间分开，各走一个branch来实现。
4. 对于基于RCAN的SR网络，可以考虑给一个RG中每个RCAB出来的feature map都作为输入进入一个multi-head self-attention模块，并且对于每一个feature map用不同的conv kernel size。这样就在每一个RG的最后加入了一个multi-head self-attention模块。
5. 对于基于RCAN的SR网络，可以考虑给每个RG出来的feature map都作为输入进入multi-head self-attention模块，并且对于每一个feature map用不同的conv kernel size。这样就在整个网络最后加入了一个multi-head self-attention模块。
6. 对于基于U-Net的SR网络，可以考虑给每一个U-Net decoding layer中出来的feature map都作为输入进入一个multi-head self-attention模块，并且对于每一个feature map用不同的conv kernel size。这样就在整个U-Net的最后加入了一个multi-head self-attention模块。



### TTSR MRI最近还2xTTSR MRI  CA24RCAN MR 
1. ~~TTSR的LR输入，HR Reference分别过小波变换得到各自的三个高频分量进LTE，同时在TTSR最外侧加一个long skip connection从而保证LR的信息中的低频部分也都被使用。该方案最后是这样做的：apply wavelet transformation to get high freq components of lr, lrsr, ref, refsr, and search transfer for all those high freq components in freq domain. Then fuse the output soft-attention map, S, and e transferred HR texture features, T, in freq domain together with high freq components of lr MRI image, to generate sr high freq components in freq domain by using MainNet. At last step apply inverse wavelet transformation to get back sr MRI image in image domain.~~。已完成，现在需要跑这种Reference过小波的TTSR MRI方案（Wavelet TTSR MRI）的2x,4x放大，和纯TTSR MRI with CALayer的方案的2x，4x放大，然后一定要把生成的SR图与RCAN MRI SR的图比较。另外就是找一个带有T1 T2不同模态的MRI公共数据集来跑TTSR MRI SR和Wavelet TTSR MRI SR看看效果怎么样。
20210308已经跑了第一次Wavelet TTSR MRI model但居然ssim仅有0.3，这太奇怪了，不确定code是否哪里有错误还没找到，还需要再研究。
3. TTSR的LR_UP输入可以不用LR_UP而用RCAN网络生成的SR数据replace现有的LR_UP


### 其他新内容最近还要做的是
1. 研究invertable image rescaling那个网络结构和代码 给我们的MRI SR任务用



        

# 2019年的内容：
## 最近新任务
1. ~~重新生成LR图像，对2D网络图像只做层内模糊。并且需要提高生成的LR的SSIM~~
        
        已重新生成2D 1/4，1/6，1/8 LR图像。
2. 解决过拟合

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
		
		现在已经在最后一层之后带dropout,可以调大dropout prob
        
        L2 regularization。但有论文说不应该只是weight decay实现L2 regularization，而应该改这个方式。https://arxiv.org/pdf/1711.05101.pdf
		已用weight decay = e-5
	
		early stop。监测test data的loss的趋势，每次training之后都看一下test data的loss。之后需要在大概test data的loss不降却上升时候stop。
		已经设计了每次training完一个epoch后就用test data看一下对于test data的loss情况，与之前的test loss进行比较，并在test loss最低时保留当前网络且输出SR

		减小batch size也可以一定程度防止overfitting。（原理详见下文，也可以看Goodfellow'的书中有说到: Small batches can oﬀer a regularizing eﬀect (Wilson and Martinez, 2003), perhaps due to the noise they add to the learning process. Generalization error is often best for a batch size of 1. Training with such a small batch size might require a small learning rate to maintain stability because of the high variance in the estimate of the gradient. The total runtime can be very high as a result of the need to make more steps, both because of the reduced learning rate and because it takes more steps to observe the entire training set.）
		实验证明batch size为16时候防止overfitting效果不错
		
batch size和学习率如何影响网络的性能
链接：https://www.zhihu.com/question/32673260/answer/675161450
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

可以参考这篇文章：龙鹏-言有三：【AI不惑境】学习率和batchsize如何影响模型的性能？​zhuanlan.zhihu.com目前深度学习模型多采用批量随机梯度下降算法进行优化，随机梯度下降算法里n是批量大小(batchsize)，η是学习率(learning rate)。可知道除了梯度本身，这两个因子直接决定了模型的权重更新，从优化本身来看它们是影响模型性能收敛最重要的参数。学习率直接影响模型的收敛状态，batchsize则影响模型的泛化性能，两者又是分子分母的直接关系，相互也可影响，因此这一次来详述它们对模型性能的影响。模型性能对batchsize虽然没有学习率那么敏感，但是在进一步提升模型性能时，batchsize就会成为一个非常关键的参数。（关于学习率对模型性能的影响，可以参考上文）1 大的batchsize减少训练时间，提高稳定性这是肯定的，同样的epoch数目，大的batchsize需要的batch数目减少了，所以可以减少训练时间，目前已经有多篇公开论文在1小时内训练完ImageNet数据集。另一方面，大的batch size梯度的计算更加稳定，因为模型训练曲线会更加平滑。在微调的时候，大的batch size可能会取得更好的结果。2 大的batchsize导致模型泛化能力下降在一定范围内，增加batchsize有助于收敛的稳定性，但是随着batchsize的增加，模型的性能会下降，来自于文[5]。这是研究者们普遍观测到的规律，虽然可以通过一些技术缓解。这个导致性能下降的batch size在上图就是8000左右。那么这是为什么呢？研究[6]表明大的batchsize收敛到sharp minimum，而小的batchsize收敛到flat minimum，后者具有更好的泛化能力。两者的区别就在于变化的趋势，一个快一个慢，造成这个现象的主要原因是小的batchsize带来的噪声有助于逃离sharp minimum。Hoffer[7]等人的研究表明，大的batchsize性能下降是因为训练时间不够长，本质上并不少batchsize的问题，在同样的epochs下的参数更新变少了，因此需要更长的迭代次数。总之batchsize在变得很大(超过一个临界点)时，会降低模型的泛化能力。在这个临界点之下，模型的性能变换随batch size通常没有学习率敏感。3 学习率和batchsize的关系通常当我们增加batchsize为原来的N倍时，要保证经过同样的样本后更新的权重相等，按照线性缩放规则，学习率应该增加为原来的N倍[5]。但是如果要保证权重的方差不变，则学习率应该增加为原来的sqrt(N)倍[7]，目前这两种策略都被研究过，使用前者的明显居多。从两种常见的调整策略来看，学习率和batchsize都是同时增加的。学习率是一个非常敏感的因子，不可能太大，否则模型会不收敛。同样batchsize也会影响模型性能，那实际使用中都如何调整这两个参数呢？研究[8]表明，衰减学习率可以通过增加batchsize来实现类似的效果，这实际上从SGD的权重更新式子就可以看出来两者确实是等价的，文中通过充分的实验验证了这一点。研究[9]表明，对于一个固定的学习率，存在一个最优的batchsize能够最大化测试精度，这个batchsize和学习率以及训练集的大小正相关。对此实际上是有两个建议：如果增加了学习率，那么batch size最好也跟着增加，这样收敛更稳定。尽量使用大的学习率，因为很多研究都表明更大的学习率有利于提高泛化能力。如果真的要衰减，可以尝试其他办法，比如增加batch size，学习率对模型的收敛影响真的很大，慎重调整。

	除了以上一些思路之外，还有建议尝试
	扩大数据集，early stop, 正则化，Dropout，VBN, 标签平滑,特征匹配



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

18. ~~调通2D_RCAN_Based_MRI_SR，使其可以生成变大size或者不变size的SR。~~

19. 基于2D_RDN_Based_MRI_SR,2D_DDBPN_Based_MRI_SRh跟2D_RCAN_Based_MRI_SR，用一样的size的LR图像和HR图像训练，尝试各种loss等方案领其生成足够好SR图像。

20. 基于2D_RDN_Based_MRI_SR,2D_DDBPN_Based_MRI_SR还有2D_RCAN_Based_MRI_S，用小size的LR图像和大size的HR图像训练，尝试各种loss等方案领其生成足够好的大size的SR图像。之后做transfer learning，固定网络多数parameters只令最后一部分可变trainable(e.g. 最后的upscaling layer trainable)，用大size的HR图像跟更大size的SHR图像训练，让网络生成更大size的足够好的ESR(Enhanced Super Resolution)图像。

21. 基于论文Image Enhancement by Recurrently-trained Super-resolution Network (https://arxiv.org/abs/1907.11341) 提供的recurrently training 思路来生成比HR图像更牛逼的ESR(Enhanced Super Resolution)图像

22. MRI Segmentation. 好像主流应该就是用U-Net来做MRI segmentation (U-Net见 U-Net: Convolutional Networks for Biomedical Image Segmentation https://arxiv.org/abs/1505.04597). 根据jianan现在的理解，如果有标注好的训练数据，完全可以直接用RDN和DDBPN的网络结构直接做training来做segmentation。如何设计一个好的loss function可能是重点。

23. 我们有没有可能这样做，参考ESRGAN论文内提到的网络参数插值的方案，对我们的CNN的MRI SR网络也做两种导向的loss。其一是PSNR导向的比如只有图像L1或者MSE的loss，另一种loss可以比如SSIM,gradient的L1 MSE的loss，训练俩sets网络参数，然后对参数插值得到一个新set参数，生成SR。感觉这个就是channel attention的类似思路，只是插值所用的具体比例人工给定而不是网络训练得出。现在看到的情况是当把各种loss求和加在一块做总loss时，gradient L1 loss降不下去，也许当只将gradient L1 loss作为目标loss时候更容易降低

24. Ensemble assisted deep learning。还有另外一种思路，我们设计了3种不同方案来实现multi-task&loss orientation networks。先搞方案2，再弄1，之后3（1+2的comb network）

25. AdamW.据说应该是把pytorch中原版的Adam的bug修复了，将learning rate和weight decay解耦合(https://www.zhihu.com/question/67335251/answer/262989932)。pytorch中目前在Adam下的L2 regularization的实现方法如下:在optimizer之前加上L2 regularization项来通过weight decay实现L2正则, 这样L2正则的作用就受到了优化器和learning rate(alpha)的影响(当然，这个是L2正则的正确实现方案). Adam的自适应归一化, 将梯度大的weight也进行了归一化, 于是抵消了weight decay的正则化的作用. 使梯度值较大的weight, 下降的比预想的要少. 换句话说, 两个weight一样大, weight decay对他们应该起到的作用是一样的. 但是其中梯度比较大的那个因为Adam的归一化, 反而下降的比较小. 也就表明当learning rate和weight decay耦合情况下，L2正则和weight decay并非完全等价. 而AdamW optimizer将learning rate和weight decay解耦合，从而令weight decay的regularization效用不会被learning rate影响所减小，于是正确的实现weight decay regularization。
文章在这 https://arxiv.org/pdf/1711.05101.pdf  代码在这  https://github.com/mpyrozhok/adamwr/blob/master/adamw.py

		用法:
		    batch_size = 32
		    epoch_size = 1024
		    model = resnet()
		    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
		    for epoch in range(100):
			optimizer.step()
			train_for_every_batch(...)
			    ...
			    optimizer.zero_grad()
			    loss.backward()
			    optimizer.step()
			    optimizer.batch_step()
			validate(...)



25. AdaBound. 几个月前的新optimizer，据说很不错。可以试一试。文章在这 https://www.luolc.com/publications/adabound/ 代码在这 https://github.com/Luolc/AdaBound
	
		An optimizer that trains as fast as Adam and as good as SGD, for developing state-of-the-art deep learning models on a wide variety of popular tasks in the field of CV, NLP, and etc.
		用法: optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
		AdaBound is an optimizer that behaves like Adam at the beginning of training, and gradually transforms to SGD at the end. The final_lr parameter indicates AdaBound would transforms to an SGD with this learning rate. In common cases, a default final learning rate of 0.1 can achieve relatively good and stable results on unseen data. It is not very sensitive to its hyperparameters. See Appendix G of the paper for more details.
		
26. RAdam. 最新optimizer,，据说很不错。可以试一试。文章在这 https://arxiv.org/abs/1908.03265v1 代码在这 https://github.com/LiyuanLucasLiu/RAdam

27. Rank loss. To address the problem, we propose Super-Resolution Generative Adversarial Networks with Ranker (RankSRGAN) to optimize generator in the direction of perceptual metrics. Specifically, we first train a Ranker which can learn the behavior of perceptual metrics and then introduce a novel rank-content loss to optimize the perceptual quality. The most appealing part is that the proposed method can combine the strengths of different SR methods to generate better results.
		
		论文 https://wenlongzhang0724.github.io/Projects/RankSRGAN#RankSRGAN
		代码 https://github.com/chisyliu/RankSRGAN
增加了一个网络Ranker。这个Ranker是一个类似VGG的网络，功能是给输入的图片打分。通过输入不同网络生成的超轻图片进行与训练，然后在训练G网络时，R输出的分数作为G网络loss的一部分。

31. LR图片生成方法：
- k空间边缘部分置零，可以在y方向上，模拟实际图像采集时的降采样，生成与HR图片相同尺寸的模糊图片。用来训练网络从降采样信号中重建原图片。
- 裁剪k空间边缘部分，生成按比例缩小的图片，可以用来训练放大图片的网络。
- k空间边缘同时在x和y方向上置零，生成与HR图片尺寸相同的模糊图片。应该可以与上一条有相同的应用。
- 卷积高斯分布卷积核，然后相邻像素取均值。这种方法不能模拟MRI实际的成像方式。

32. 根据ESRGAN论文，其有一个改动是关于perception loss的。该论文建议利用VGG的激活层前的特征改善感知损失，会使得生成的图像有更加清晰的边缘（为亮度的一致性和纹理恢复提供更强的监控。 我们是否已经这么做啦？

		我们提出在激活层之前使用特性，这将克服原始设计的两个缺点。第一，被激活的特征是非常稀疏的，特别是在非常深的网络之后，如图6所示。例如，图像“baboon”激活神经元的平均百分率在VGG19-54层后仅为11.17%（我们使用预先训练的19层VGG网络[37]，其中54表示5^{th}最大池化层之前通过4^{th}卷积获得的特征，表示高级特征，类似地，22表示低级特征）。稀疏激活提供弱的监督，从而导致性能较差。第二，使用激活后的特征也会造成重建后的图像亮度与真实图像不一致
		
我们也可以考虑做同样的事情。
但是 根据其代码 https://github.com/chisyliu/BasicSR/blob/master/codes/models/modules/discriminator_vgg_arch.py
和 https://github.com/chisyliu/BasicSR/blob/master/codes/models/networks.py
完全看不出来按照论文建议的方案这样做了。我们需要研究一下代码应该怎样写。

33. 看下这俩篇paper，做一下这俩种方案

		Some recently proposed advanced methods:
			•SRNTT. 2019. Image Super-Resolution by Neural Texture Transfer
				oPaper: https://arxiv.org/pdf/1903.00834.pdf
				oComment: Amazing! It seems the reconstructed SR image from this approach has much better resolution than the SR image from SRGAN.
			
			•2019. ODE-inspired Network Design for Single Image Super-Resolution
				oPaper: http://openaccess.thecvf.com/content_CVPR_2019/papers/He_ODE-Inspired_Network_Design_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf
				oComment: It seems the approach proposed is somehow change in the ResNet block, it is NOT hard to implement and easy to be incorporated into any approaches we use.

34. 这个还没有published的论文设计的提高动画的resolution，设计了一种专门针对图像中任务的边缘信息做优化提高resolution的算法，该算法逻辑如下:因为任意图像可以分解为low freq component(represents for texture information)与high freq component(represents for edge information)。当图像resolution越大时，分解后得出的high freq component对应的图像域表征的edge则应越细。于是该算法就是在minimize edge thickness。我们可以研究一下看看是否可以融合到我们的SR或者De-MotionArtifact任务。
	
		 论文 https://github.com/chisyliu/Anime4K/blob/master/Preprint.md
	 
		 代码 https://github.com/chisyliu/Anime4K
		 
	 	Comment 1: https://kknews.cc/comic/5rzggal.html
	 
	 	Comment 2: https://www.oschina.net/p/anime4k
	 
	 	Comment 3: https://blog.csdn.net/hahabeibei123456789/article/details/100007662
		
35. 从这个34.动画super resolution项目得到的两个可能有助于我们MRI SR的edge信息恢复的idea，分别做一下。方案如下，定义Edge Quality Loss
		 
		1. Edge Quality Loss option1: FFT之后的k space loss应该可以分解成为low freq component loss(represents for texture information)与high freq component loss(represents for edge information)，我们可以考虑给high freq componenet loss更大的weight从而加强edge的恢复效果
		
		2. Edge Quality Loss option2: 根据34.动画super resolution项目的逻辑，假如图像resolution越大则该图像的high freq component对应的图像域表征的edge则应越细(根据我们的理解，除了细以外且应该越亮或者说其强度越强)，于是可以考虑将k space loss的high freq component loss做IFFT回到图像域，minimize其对应的edge information 所占的number of pixels并且同时maximize相应的总强度。这里的问题是，如果数学上定义这个minmax并且coding。
		
现阶段对Edge Quality Loss option 2有几种思路

	2.1. 把SR和HR各自的k space高频部分做IFFT回image field后，强度过threshold的pixel(我们认为设定合理的强度threshold，超过该threshold的就是edge information)的总数求出来，之后用两个数量相减的L1或者之类的criteria作为loss。虽然逻辑上即使SR中表征edge informationd的所有pixel数量接近HR中表征edge informationd的所有pixel的数量也不能表示SR的edge information的图案分布和HR的edge information的图案分布接近，但是由于MRI SR问题中只有一个object，并且由于Input还有LR所以SR和HR即使在training的开始各自的edge information图案分布也不会差太远，所以这样简单的minimize"SR中表征edge informationd的所有pixel数量和HR中表征edge informationd的所有pixel的数量的差"其实有可能得到还可以的结果。
	
	2.2. 把SR和HR各自的k space高频部分做IFFT回image field后，强度过threshold的pixel(我们认为设定合理的强度threshold，超过该threshold的就是edge information)的所有坐标(image中的index信息)与总数求出来，理论上SR(特别是在training开始之时)的表征edge information的pixel应该数量比HR的表征edge information的pixel的数量多(因为LR的edge比HR的edge thick)。然后从HR的表征edge information的一个pixel开始，对其每一个pixel在所有的SR的的表征edge information的pixels中找到距离最近的pixel其求距离，并且在对SR的的表征edge information所有的pixels删掉该pixel。对HR的表征edge information所有的pixelx重复上述动作，直到HR的表征edge information所有的pixelx都找到了一个"match"的neighbor(这其实就是NN algorithm)。然后再把这些distance一起加上"剩下的SR的表征edge information的每个pixel和0的距离"作为loss。
	
	2.3. Wasserstein distance是否可能用在刻画SR的edge information与HR的edge information的距离？如果可以(比如可以描述这两个形状的相似程度)，怎样用？
	
	2.4. 是否有什么criteria能用于刻画两个形状的相似程度？如果有，怎样联系到pixel level的index(position information in image fiedl)


36. 新Backbone可以替换ResNet主要的Residual block模块

		Res2Net: A New Multi-scale Backbone Architecture
		a.Paper: https://arxiv.org/abs/1904.01169
		b.Code: https://github.com/chisyliu/Res2Net
		c.Code: https://github.com/chisyliu/Res2Net-1
		d.Comment 1: https://zhuanlan.zhihu.com/p/61407825
		e.Comment 2: https://xueqiu.com/3426965578/124511338
		f.Comment 3: https://www.chainnews.com/articles/735829732142.htm
		g.Comment 4: https://www.jishuwen.com/d/pbYi


37. 也许multi-scale feature maps可以帮助我们生产效果更好的SR？但这只是一个猜测，需要找到合理的解释来给出某些直觉上的解释论证为什么multi-scale feature map可以生产效果更强的SR。

如何生成multi-scale feature maps?可以借助FPN或者Dilated conv。可以参考的paper如下（注意，对于Dilated conv for MRI SR主要读paper看他怎么分析 为什么Dilated conv可以提高SR效果，如何提高的）

		Compressed Sensing MRI via a Multi-scale Dilated Residual Convolution Network
		a.Paper: https://arxiv.org/abs/1906.05251




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
6. Based on RCAN, discuss the effect of different loss function:
   - MSE
   - L1
   - L1 + k-space
   - L1 + k-space + VGG
   - L1 + k-space + ssim (+ VGG)
   - L1 + k-space + gradient (+ VGG)
   - L1 + k-space + ssim + gradient (+ VGG)
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
 
