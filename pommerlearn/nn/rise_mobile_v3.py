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
from torch.nn import Sequential, Conv2d, BatchNorm2d, Hardsigmoid, Hardswish, Module
from nn.builder_util import get_act, MixConv, _ValueHead, _PolicyHead, _Stem, get_se


class _PreactResidualMixConvBlock(Module):

    def __init__(self, channels, channels_operating, kernels=None, act_type='relu', se_ratio=4, se_type="se", bn_mom=0.9):
        """
        Returns a residual block without any max pooling operation
        :param channels: Number of filters for all CNN-layers
        :param name: Name for the residual block
        :param act_type: Activation function to use
        :param se_ratio: Squeeze excitation ratio
        :param use_se: Boolean if a squeeze excitation module will be used
        :param se_type: Squeeze excitation module type. Available [None, "se", "cbam", "ca_se", "cm_se", "sa_se", "sm_se"]
        :return: symbol
        """
        super(_PreactResidualMixConvBlock, self).__init__()

        self.body = Sequential(BatchNorm2d(momentum=bn_mom, num_features=channels),
                               Conv2d(in_channels=channels, out_channels=channels_operating, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               MixConv(in_channels=channels_operating, out_channels=channels_operating, kernels=kernels),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels, kernel_size=(1, 1), bias=False),
                               )

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return x + self.body(x)


class _BottlekneckResidualBlock(Module):

    def __init__(self, channels, channels_operating, kernel=3, act_type='relu', se_type=None, bn_mom=0.9):
        """
        Returns a residual block without any max pooling operation
        :param channels: Number of filters for all CNN-layers
        :param name: Name for the residual block
        :param act_type: Activation function to use
        :param se_type: Squeeze excitation module that will be used
        :return: symbol
        """
        super(_BottlekneckResidualBlock, self).__init__()

        self.se_type = se_type
        if se_type:
            self.se = get_se(se_type=se_type, channels=channels, use_hard_sigmoid=False)
        self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels_operating, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels_operating, kernel_size=(kernel, kernel), padding=(kernel // 2, kernel // 2), bias=False, groups=channels_operating),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.se_type:
            return x + self.body(self.se(x))
        return x + self.body(x)


class _StridedBottlekneckBlock(Module):

    def __init__(self, channels, channels_operating, kernel=3, act_type='relu', se_type=None, bn_mom=0.9):
        """
        Returns a residual block without any max pooling operation
        :param channels: Number of filters for all CNN-layers
        :param name: Name for the residual block
        :param act_type: Activation function to use
        :param se_type: Squeeze excitation module that will be used
        :return: symbol
        """
        super(_StridedBottlekneckBlock, self).__init__()
        self.use_se = se_type
        if se_type:
            self.se = get_se(se_type=se_type, channels=channels, use_hard_sigmoid=False)
        self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels_operating, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels_operating,
                                      kernel_size=(kernel, kernel), stride=(2, 2), padding=(kernel // 2, kernel // 2), bias=False, groups=channels_operating),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_operating),
                               get_act(act_type),
                               Conv2d(in_channels=channels_operating, out_channels=channels*2, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels*2))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.use_se:
            return self.body(self.se(x))
        return self.body(x)


class RiseV3(Module):

    def __init__(self, channels=128, nb_input_channels=18, channels_operating=256, act_type='relu',
                 channels_value_head=1, channels_policy_head=81, value_fc_size=256,
                 select_policy_from_plane=False, kernels=None, n_labels=6, se_ratio=4,
                 se_types=None, use_avg_features=False, use_raw_features=False, value_nb_hidden=7,
                 value_fc_size_hidden=256, value_dropout=0.15, use_more_features=False, bn_mom=0.9,
                 board_height=11, board_width=11, use_downsampling=True,
                 ):
        """
        RISEv3 architecture
        :param channels: Main number of channels
        :param channels_operating: Initial number of channels at the start of the net for the depthwise convolution
        :param channel_expansion: Number of channels to add after each residual block
        :param act_type: Activation type to use
        :param channels_value_head: Number of channels for the value head
        :param value_fc_size: Number of units in the fully connected layer of the value head
        :param channels_policy_head: Number of channels for the policy head
        :param dropout_rate: Droput factor to use. If 0, no dropout will be applied. Value must be in [0,1]
        :param grad_scale_value: Constant scalar which the gradient for the value outputs are being scaled width.
                                (0.01 is recommended for supervised learning with little data)
        :param grad_scale_policy: Constant scalar which the gradient for the policy outputs are being scaled width.
        :param select_policy_from_plane: True, if policy head type shall be used
        :param kernels: List of kernel sizes used for the residual blocks. The length of the list corresponds to the number
        of residual blocks.
        :param n_labels: Number of policy target labels (used for select_policy_from_plane=False)
        :param se_ratio: Reduction ration used in the squeeze excitation module
        :param se_types: List of squeeze exciation modules to use for each residual layer.
         The length of this list must be the same as len(kernels). Available types:
        - "se": Squeeze excitation block - Hu et al. - https://arxiv.org/abs/1709.01507
        - "cbam": Convolutional Block Attention Module (CBAM) - Woo et al. - https://arxiv.org/pdf/1807.06521.pdf
        - "ca_se": Same as "se"
        - "cm_se": Squeeze excitation with max operator
        - "sa_se": Spatial excitation with average operator
        - "sm_se": Spatial excitation with max operator
        :param use_avg_features: If true the value head receives the avg of the each channel of the original input
        :param use_raw_features: If true the value receives the raw features of the pieces positions one hot encoded
        :param value_nb_hidden: Number of hidden layers of the vlaue head
        :return: symbol
        """
        super(RiseV3, self).__init__()

        self.use_raw_features = use_raw_features

        if len(kernels) != len(se_types):
            raise Exception(f'The length of "kernels": {len(kernels)} must be the same as'
                            f' the length of "se_types": {len(se_types)}')

        valid_se_types = [None, "se", "cbam", "eca_se", "ca_se", "cm_se", "sa_se", "sm_se"]
        for se_type in se_types:
            if se_type not in valid_se_types:
                raise Exception(f"Unavailable se_type: {se_type}. Available se_types include {se_types}")

        res_blocks = []
        expansion_factor = 1
        for idx, cur_kernel in enumerate(kernels):
            if use_downsampling and idx >= len(kernels) / 2:
                use_downsampling = False
                res_blocks.append(_StridedBottlekneckBlock(channels=channels, channels_operating=channels_operating*2,
                                                           kernel=cur_kernel, act_type=act_type,
                                                           se_type=se_types[idx], bn_mom=bn_mom))
                expansion_factor = 2
                board_width = round(board_width * 0.5)
                board_height = round(board_height * 0.5)
            else:
                res_blocks.append(_BottlekneckResidualBlock(channels=channels*expansion_factor, channels_operating=channels_operating*expansion_factor,
                                                            kernel=cur_kernel, act_type=act_type,
                                                            se_type=se_types[idx], bn_mom=bn_mom))

        self.body = Sequential(
            _Stem(channels=channels, bn_mom=bn_mom, act_type=act_type, nb_input_channels=nb_input_channels),
            *res_blocks,
        )
        # create the two heads which will be used in the hybrid fwd pass
        self.value_head = _ValueHead(board_height, board_width, channels*expansion_factor, channels_value_head, value_fc_size, bn_mom, act_type)
        self.policy_head = _PolicyHead(board_height, board_width, channels*expansion_factor, channels_policy_head, n_labels,
                                                bn_mom, act_type, select_policy_from_plane)

    def forward(self, x):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.body(x)
        if self.use_raw_features:
            value = self.value_head(out, x)
        else:
            value = self.value_head(out)

        policy = self.policy_head(out)

        return [value, policy]
