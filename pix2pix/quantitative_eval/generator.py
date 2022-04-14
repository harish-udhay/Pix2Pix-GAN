import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_filters=64, use_dropout=False):
        super(Generator, self).__init__()

        ub1 = SkipConnection(num_filters * 8, num_filters * 8, innermost_layer=True, input_channels=None, submodule=None, norm_layer=nn.BatchNorm2d) 

        ub2 = SkipConnection(num_filters * 8, num_filters * 8, input_channels=None, submodule=ub1, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        ub3 = SkipConnection(num_filters * 8, num_filters * 8, input_channels=None, submodule=ub2, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        ub4 = SkipConnection(num_filters * 8, num_filters * 8, input_channels=None, submodule=ub3, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)

        ub5 = SkipConnection(num_filters * 4, num_filters * 8, input_channels=None, submodule=ub4, norm_layer=nn.BatchNorm2d)
        ub6 = SkipConnection(num_filters * 2, num_filters * 4, input_channels=None, submodule=ub5, norm_layer=nn.BatchNorm2d)
        ub7 = SkipConnection(num_filters, num_filters * 2, input_channels=None, submodule=ub6, norm_layer=nn.BatchNorm2d)
        
        self.model = SkipConnection(output_channels, num_filters,outermost_layer=True, input_channels=input_channels, submodule=ub7, norm_layer=nn.BatchNorm2d)  

    def forward(self, input):
        return self.model(input)

class SkipConnection(nn.Module):
    def __init__(self, output_channels, inner_channels, input_channels=None, submodule=None, outermost_layer=False, innermost_layer=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SkipConnection, self).__init__()
        self.outermost_layer = outermost_layer
        if input_channels is None:
            input_channels = output_channels
        down_sampling_conv = nn.Conv2d(input_channels, inner_channels, kernel_size=4,
                             stride=2, padding=1, bias=False)
        down_sampling_relu = nn.LeakyReLU(0.2, True)
        down_sampling_norm = norm_layer(inner_channels)
        up_sampling_relu = nn.ReLU(True)
        up_sampling_norm = norm_layer(output_channels)

        if outermost_layer:
            up_sampling_conv = nn.ConvTranspose2d(inner_channels * 2, output_channels, kernel_size=4, stride=2, padding=1)
            downSample = [down_sampling_conv]
            upSample = [up_sampling_relu, up_sampling_conv, nn.Tanh()]
            model = downSample + [submodule] + upSample
        
        elif innermost_layer:
            up_sampling_conv = nn.ConvTranspose2d(inner_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False)
            downSample = [down_sampling_relu, down_sampling_conv]
            upSample = [up_sampling_relu, up_sampling_conv, up_sampling_norm]
            model = downSample + upSample
        
        else:
            up_sampling_conv = nn.ConvTranspose2d(inner_channels * 2, output_channels, kernel_size=4, stride=2, padding=1, bias=False)
            downSample = [down_sampling_relu, down_sampling_conv, down_sampling_norm]
            upSample = [up_sampling_relu, up_sampling_conv, up_sampling_norm]

            if use_dropout:
                model = downSample + [submodule] + upSample + [nn.Dropout(0.5)]
            else:
                model = downSample + [submodule] + upSample

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.outermost_layer:
            return self.model(input)
        else:   # add skip connections
            return torch.cat([input, self.model(input)], 1)
