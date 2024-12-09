import torch
import torch.nn as nn
import torch.nn.functional as F


    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """

        Initializes the DoubleConv module.


        This module consists of two convolutional layers, each followed by batch normalization

        and a ReLU activation function.


        Parameters:

            in_channels (int): Number of input channels.

            out_channels (int): Number of output channels.

            mid_channels (int, optional): Number of channels in the middle convolution. 

                                           If None, it will be set to out_channels.

        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
   
    def forward(self, x):
        """

        Forward pass through the DoubleConv module.


        Parameters:

            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size,

                              C is the number of channels, H is the height, and W is the width.


        Returns:

            torch.Tensor: Output tensor after applying the double convolution.

        """
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        """

        Initializes the Down module.


        This module performs downscaling using max pooling followed by a double convolution.


        Parameters:

            in_channels (int): Number of input channels.

            out_channels (int): Number of output channels.

        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """

        Forward pass through the Down module.


        Parameters:

            x (torch.Tensor): Input tensor.


        Returns:

            torch.Tensor: Output tensor after downscaling and convolution.

        """
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """Upscaling followed by a double convolution."""
    def __init__(self, in_channels, out_channels):
        """

        Initializes the Up module.


        This module performs upscaling using transposed convolution followed by a double convolution.


        Parameters:

            in_channels (int): Number of input channels.

            out_channels (int): Number of output channels.

        """
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        """

        Forward pass through the Up module.


        Parameters:

            x1 (torch.Tensor): Input tensor from the previous layer.

            x2 (torch.Tensor): Corresponding tensor from the downsampling path.


        Returns:

            torch.Tensor: Output tensor after upscaling and convolution.

        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer."""
    def __init__(self, in_channels, out_channels):
        """

        Initializes the OutConv module.


        This module consists of a single convolutional layer that reduces the number of channels

        to the desired output classes.


        Parameters:

            in_channels (int): Number of input channels.

            out_channels (int): Number of output channels (number of classes).

        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """

        Forward pass through the OutConv module.


        Parameters:

            x (torch.Tensor): Input tensor.


        Returns:

            torch.Tensor: Output tensor after applying the output convolution.

        """
        return self.conv(x)
    

class UNet(nn.Module):
    """

    U-Net architecture for image segmentation.


    The U-Net model consists of a contracting path (downsampling) and an expansive path (upsampling),

    allowing for precise localization while maintaining context. It is particularly effective for

    biomedical image segmentation.


    Attributes:

        n_channels (int): Number of input channels (e.g., 1 for grayscale images, 3 for RGB images).

        n_classes (int): Number of output classes for segmentation.

        inc (DoubleConv): Initial double convolution layer.

        down1 (Down): First downsampling layer.

        down2 (Down): Second downsampling layer.

        down3 (Down): Third downsampling layer.

        bottom (DoubleConv): Bottom layer of the U-Net.

        up1 (Up): First upsampling layer.

        up2 (Up): Second upsampling layer.

        up3 (Up): Third upsampling layer.

        up4 (Up): Fourth upsampling layer.

        outc (OutConv): Output convolution layer to produce the final segmentation map.

    """
    def __init__(self, n_channels, n_classes):
        """

        Initializes the U-Net model.


        Parameters:

            n_channels (int): Number of input channels (e.g., 1 for grayscale images, 3 for RGB images).

            n_classes (int): Number of output classes for segmentation.

        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bottom = DoubleConv(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """

        Forward pass through the U-Net model.


        This method defines the forward computation of the U-Net architecture, which includes

        the downsampling and upsampling paths, as well as the final output layer.


        Parameters:

            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size,

                              C is the number of input channels, H is the height, and W is the width.


        Returns:

            torch.Tensor: Output tensor of shape (N, n_classes, H_out, W_out), where H_out and W_out

                          are the height and width of the output segmentation map.

        """
        x1 = self.inc(x)          # Initial convolution

        x2 = self.down1(x1)      # First downsampling

        x3 = self.down2(x2)      # Second downsampling

        x4 = self.down3(x3)      # Third downsampling

        x5 = self.bottom(x4)     # Bottom layer

        x = self.up1(x5, x4)     # First upsampling

        x = self.up2(x, x3)      # Second upsampling

        x = self.up3(x, x2)      # Third upsampling

        x = self.up4(x, x1)      # Fourth upsampling

        logits = self.outc(x)     # Output layer
        return logits
    
