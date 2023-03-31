# DALLE 

## **dVAE**

![img](https://pic1.zhimg.com/v2-5baa6c8373d651ed63142b9c205f2468_r.jpg)

vae

![img](https://pic1.zhimg.com/80/v2-f37a8f00da3b1edf29643d998f311494_720w.webp)

vq-vae

dvae和 VQVAE 方法相似，dVAE 的 encoder 是将图像的 patch 映射到 8192 的词表中，**论文中将其分布设为在词表向量上的均匀分类分布，这是一个离散分布，由于不可导的问题，此时不能采用重参数技巧**，DALL・E 使用 **Gumbel Softmax trick 来解决这个问题** (简单来说就是 arg max 不可导，可以用 softmax 来近似代替 max，而 arg softmax 是可导的，后面补一篇 Gumbel Softmax 的文章)。

## Clip

![img](https://pic1.zhimg.com/v2-65a6d3fe23b60a89a2d4e41a902a5ab8_r.jpg)

 CLIP 的推理过程：

1. 先将预训练好的 CLIP 迁移到下游任务，如图 (2) 所示，先将下游任务的标签构建为一批带标签的文本 (例如 A photo of a {plane})，然后经过 Text Encoder 编码成一批相应的 word embedding。

2. 然后将没有见过的图片进行 zero-shot 预测，如图 (3) 所示，通过 Image Encoder 将一张小狗的图片编码成一个 feature embedding，然后跟 (2) 编码的一批 word embedding 先归一化然后进行点积，最后得到的 logits 中数值最大的位置对应的标签即为最终预测结果。

在 DALL・E 中，CLIP 的用法跟上述过程相反，提供输入文本和一系列候选图片，先通过 Stage One 和 Stage Two 生成文本和候选图片的 embedding，然后通过文本和候选图片的匹配分数进行排序，最后找到跟文本最匹配的图片。

![img](https://pic4.zhimg.com/v2-68aee588111332bc36975912295e622f_r.jpg)



DALL・E 的目标是把文本 token 和图像 token 当成一个数据序列，通过 Transformer 进行自回归。由于图片的分辨率很大，如果把单个 pixel 当成一个 token 处理，会导致计算量过于庞大，于是 DALL・E 引入了一个 dVAE 模型来降低图片的分辨率。

DALL・E 的整体流程如下：

- 第一个阶段，先训练一个 dVAE 把每张 256x256 的 RGB 图片压缩成 32x32 的图片 token，每个位置有 8192 种可能的取值 (也就是说 dVAE 的 encoder 输出是维度为 32x32x8192 的 logits，然后通过 logits 索引 codebook 的特征进行组合，codebook（vq-vae） 的 embedding 是可学习的)。
- 第二阶段，用 BPE Encoder 对文本进行编码，得到最多 256 个文本 token，token 数不满 256 的话 padding 到 256，然后将 256 个文本 token 与 1024 个图像 token 进行拼接，得到长度为 1280 的数据，最后将拼接的数据输入 Transformer 中进行自回归训练。
-  推理阶段，给定一张候选图片和一条文本，通过 transformer 可以得到融合后的 token，然后用 dVAE 的 decoder 生成图片，最后通过预训练好的 CLIP 计算出文本和生成图片的匹配分数，采样越多数量的图片，就可以通过 CLIP 得到不同采样图片的分数排序。

从以上流程可知，**dVAE**、**Transformer** 和 **CLIP** 三个模型都是不同阶段独立训练的。下面讲一下 dVAE、Transformer 和 CLIP 三个部分。