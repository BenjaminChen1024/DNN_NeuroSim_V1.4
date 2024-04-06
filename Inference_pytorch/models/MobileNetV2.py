import torch
from torch import nn, Tensor

from modules.quantization_cpu_np_infer import QConv2d, QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear

from torchvision.models._utils import _make_divisible
from torch.hub import load_state_dict_from_url

name=0
__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "mobilenet_v2"]

model_urls = {
    'mobilenetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',}

def Conv2d(in_planes, out_planes, kernel_size, stride, padding, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":        
        conv2d = QConv2d(in_planes, out_planes, kernel_size, stride, padding, logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                         wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                         name = 'Conv'+'_'+str(name)+'_', model = args.model, parallelRead=args.parallelRead)
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False,
                         logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, cuda=args.cuda,
                         name = 'Conv'+'_'+str(name)+'_' )
    name += 1
    return conv2d


def Linear(in_planes, out_planes, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":
        linear = QLinear(in_planes, out_planes, 
                        logger=logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, 
                        name='FC'+'_'+str(name)+'_', model = args.model, parallelRead=args.parallelRead)
    elif args.mode == "FP":
        linear = FLinear(in_planes, out_planes, bias=False,
                         logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, cuda=args.cuda,
                         name='FC'+'_'+str(name)+'_')
    name += 1
    return linear

def Conv2dNormActivation(in_channels, out_channels, kernel_size=3, stride=1, padding=None, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, args=None, logger=None):

    layers = [Conv2d(in_channels, out_channels, kernel_size, stride, padding, args=args, logger=logger)]
    
    if norm_layer is not None:
            layers.append(norm_layer(out_channels))

    if activation_layer is not None:
            layers.append(activation_layer())

    return layers


# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer = None, args=None, logger=None):
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.extend(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, 
                                     norm_layer=norm_layer, activation_layer=nn.ReLU6, 
                                     args=args, logger=logger)

            )
        layers.extend(
                # dw
                Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride, 
                                     norm_layer=norm_layer, activation_layer=nn.ReLU6, 
                                     args=args, logger=logger)
        )
        layers.extend(
            [

                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, args=args, logger=logger),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, 
                 round_nearest=8, block=None, norm_layer=None, dropout=0.2, args=None, logger=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = Conv2dNormActivation(3, input_channel, stride=2, 
                                 norm_layer=norm_layer, activation_layer=nn.ReLU6, 
                                 args=args, logger=logger)
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, args=args, logger=logger))
                input_channel = output_channel
        # building last several layers
        features.extend(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, 
                norm_layer=norm_layer, activation_layer=nn.ReLU6, 
                args=args, logger=logger)
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            Linear(self.last_channel, num_classes, args=args, logger=logger),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, QConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, QLinear)):
                nn.init.constant_(m.weight, 0)
                # nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
    
def _mobilenet(arch, block, pretrained=None, progress=True, args=None, logger=None):
    model = MobileNetV2(block=block, args=args, logger=logger)
    print(model)
    if pretrained==True:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    elif pretrained is not None:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def mobilenet_v2(pretrained=None, progress=True, args=None, logger=None):
    return _mobilenet('mobilenet_v2', InvertedResidual, pretrained, progress, args=args, logger=logger)