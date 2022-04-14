from torch import nn

def conv_norm_relu(n_in, n_out, transpose=False, **kwargs):
    """Standard convolution -> instance norm -> relu block.
    
    Params:
        n_in -- number of input channels
        n_out -- number of filters/output channels
        transpose -- whether to use a egular or transposed convolution layer
        kwargs -- other args passed to the convolution layer
    """
    if transpose:
        conv = nn.ConvTranspose2d(n_in, n_out, bias=True, **kwargs)
    else:
        conv = nn.Conv2d(n_in, n_out, bias=True, **kwargs)
    return [conv, nn.InstanceNorm2d(n_out), nn.ReLU(True)]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding = 0, padding_mode = "reflect", bias = True),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels, in_channels, 3, padding = 0, padding_mode = "reflect", bias = True),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, input):
        residual = self.residual_block(input)
        residual += input[..., 2:-2, 2:-2]  # apply skip-connection
        return residual

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        num_residual_blocks = 9
        res_blocks = [ResidualBlock(256) for _ in range(num_residual_blocks)]
        dropouts = [nn.Dropout(inplace=True) for _ in range(num_residual_blocks)]        
        self.generator = nn.Sequential(
            #c7s1-64
            nn.Conv2d(3, 64, 7, padding = 2, padding_mode = "reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            #d128
            *self.convolution_layer(in_channels = 64, 
                                    out_channels = 128, 
                                    kernel_size = 3, 
                                    instance_norm = 128, 
                                    stride = 2, 
                                    padding = 1
                                ),
            
            #d256
            *self.convolution_layer(in_channels = 128, 
                                    out_channels = 256, 
                                    kernel_size = 3, 
                                    instance_norm = 256, 
                                    stride = 2, 
                                    padding = 1
                                ),
            nn.ReflectionPad2d(18),
            *[x for pair in zip(res_blocks, dropouts) for x in pair],
            
            #u128
            *self.transposed_convolution_layer( in_channels = 256, 
                                                out_channels = 128, 
                                                kernel_size = 3, 
                                                instance_norm = 128, 
                                                stride = 2, 
                                                padding = 1, 
                                                output_padding = 1
                                            ),
            
            #u64
            *self.transposed_convolution_layer( in_channels = 128, 
                                                out_channels = 64, 
                                                kernel_size = 3, 
                                                instance_norm = 64, 
                                                stride = 2, 
                                                padding = 1, 
                                                output_padding = 1
                                            ),
            
            #c7s1-3
            nn.Conv2d(64, 3, 7, padding = 3, padding_mode = "reflect"),
            nn.InstanceNorm2d(3),
            nn.Tanh()
        )
    
    @staticmethod
    def convolution_layer(in_channels, out_channels, kernel_size, instance_norm, stride = 1, padding = 0):
        return(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode = "reflect", bias=True),
            nn.InstanceNorm2d(instance_norm),
            nn.ReLU(inplace = True),
        )

    @staticmethod
    def transposed_convolution_layer(in_channels, out_channels, kernel_size, instance_norm, stride = 1, padding = 0, output_padding = 0):
        return(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True),
            nn.InstanceNorm2d(instance_norm),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.generator(x)
