"""
Upgrade of RISE architecture using mixed depthwise convolutions, preactivation residuals and dropout
 proposed by Johannes Czech:

Influenced by the following papers:
    * MixConv: Mixed Depthwise Convolutional Kernels, Mingxing Tan, Quoc V. Le, https://arxiv.org/abs/1907.09595
    * ProxylessNas: Direct Neural Architecture Search on Target Task and Hardware, Han Cai, Ligeng Zhu, Song Han.
     https://arxiv.org/abs/1812.00332
    * MnasNet: Platform-Aware Neural Architecture Search for Mobile,
     Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le
     http://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html
    * FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search,
    Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, Kurt Keutzer,
    http://openaccess.thecvf.com/content_CVPR_2019/html/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.html
    * MobileNetV3: Searching for MobileNetV3,
    Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam.
    https://arxiv.org/abs/1905.02244
    * Convolutional Block Attention Module (CBAM),
    Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
    https://arxiv.org/pdf/1807.06521.pdf

"""
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh, Linear, Hardsigmoid, Hardswish


def preact_residual_dmixconv_block(data, channels, channels_operating, name, kernels=None, act_type='relu',
                                   se_ratio=4, se_type="se"):
    """
    Returns a residual block without any max pooling operation
    :param data: Input data
    :param channels: Number of filters for all CNN-layers
    :param name: Name for the residual block
    :param act_type: Activation function to use
    :param se_ratio: Squeeze excitation ratio
    :param use_se: Boolean if a squeeze excitation module will be used
    :param se_type: Squeeze excitation module type. Available [None, "se", "cbam", "ca_se", "cm_se", "sa_se", "sm_se"]
    :return: symbol
    """
    # TODO
    pass
