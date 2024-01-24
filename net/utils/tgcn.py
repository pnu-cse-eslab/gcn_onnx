# # The based unit of graph convolutional networks.

# import torch
# import torch.nn as nn

# class ConvTemporalGraphical(nn.Module):

#     r"""The basic module for applying a graph convolution.

#     Args:
#         in_channels (int): Number of channels in the input sequence data
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (int): Size of the graph convolving kernel
#         t_kernel_size (int): Size of the temporal convolving kernel
#         t_stride (int, optional): Stride of the temporal convolution. Default: 1
#         t_padding (int, optional): Temporal zero-padding added to both sides of
#             the input. Default: 0
#         t_dilation (int, optional): Spacing between temporal kernel elements.
#             Default: 1
#         bias (bool, optional): If ``True``, adds a learnable bias to the output.
#             Default: ``True``

#     Shape:
#         - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
#         - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
#         - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
#         - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

#         where
#             :math:`N` is a batch size,
#             :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
#             :math:`T_{in}/T_{out}` is a length of input/output sequence,
#             :math:`V` is the number of graph nodes. 
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  t_kernel_size=1,
#                  t_stride=1,
#                  t_padding=0,
#                  t_dilation=1,
#                  bias=True):
#         super().__init__()

#         self.kernel_size = kernel_size
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels, # * kernel_size,
#             kernel_size=(t_kernel_size, 1),
#             padding=(t_padding, 0),
#             stride=(t_stride, 1),
#             dilation=(t_dilation, 1),
#             bias=bias)

#     def forward(self, x, A):
#         assert A.size(0) == self.kernel_size

#         x = self.conv(x)

#         # n, kc, t, v = x.size()
#         # x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
#         # x = torch.einsum('nkctv,kvw->nctw', (x, A))

#         return x.contiguous(), A


# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        
        self.conv2 = nn.Conv2d(
            kernel_size,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(out_channels, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        
        self.conv3 = nn.Conv2d(
            kernel_size,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(out_channels, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    # def forward(self, x, A):
    #     assert A.size(0) == self.kernel_size

    #     x = self.conv(x)

    #     n, kc, t, v = x.size()
    #     x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
#       

    #     return x.contiguous(), A

    def forward(self, x, A, usegcn=True):
        assert A.size(0) == self.kernel_size

        # print("")
        # print(x.shape)

        x = self.conv(x)

        # print(x.shape)

        n, kc, t, v = x.size()
        # print(n, kc, t,v )
        x = x.view(n, self.kernel_size, (kc//self.kernel_size) * t, v)
        _, _, w = A.size()
        # print(x.shape)
        # print(A[0].unsqueeze(0).shape)

        # print("")
        # print(x.shape)
        # print(A.unsqueeze(1).unsqueeze(0).shape)
        # print(x.unsqueeze(-1).shape)
        # print(A.unsqueeze(1).unsqueeze(1).unsqueeze(0).shape)
        # print(torch.sum(torch.mul(x.unsqueeze(-1), A.unsqueeze(1).unsqueeze(1).unsqueeze(0)), (1, 4)))
        # print(torch.einsum('nkctv,kvw->nctw', (x, A)))
        
        # x = torch.einsum('nkcv,kvw->ncw', (x, A))
        # x = torch.sum(torch.mul(x, A.unsqueeze(1).unsqueeze(0)), (1, 3))
        # print("x.shape in tgcn")
        # print(x.shape)
        if usegcn:
            #  print("result shape") 
            # print(result.shape)
            x_splits = torch.split(x, 1, 1)
            A_splits = torch.split(A.unsqueeze(0), 1, 1)
            # print(x_splits[0].shape)
            # print(A_splits[0].shape)
            # x_z     
            # b0 = torch.einsum('nckv,bkvw->nckv', x_splits[0], A_splits[0])
            # b1 = torch.einsum('nckv,bkvw->nckv', x_splits[1], A_splits[2])
            # b2 = torch.einsum('nckv,bkvw->nckv', x_splits[2], A_splits[2])
            b0 = torch.sum(torch.mul(x_splits[0].view(n, (kc//self.kernel_size) * t, v, 1), A_splits[0]), 2)
            b1 = torch.sum(torch.mul(x_splits[1].view(n, (kc//self.kernel_size) * t, v, 1), A_splits[1]), 2)
            b2 = torch.sum(torch.mul(x_splits[2].view(n, (kc//self.kernel_size) * t, v, 1), A_splits[2]), 2)
            # print(b0.shape)
            x = torch.stack((b0, b1, b2), 1)
            x = torch.sum(x, 1)
            # print(torch.einsum('nkctv,kvw->nctw', (x, A)))

            # x = torch.einsum('nkctv,kvw->nctw', (x, A))

            # print(x.shape)
        else:
            x = self.conv2(x)
            pass
        # print(x.shape)
        x = x.view(n, kc//self.kernel_size, t, v)
        
        # print(x.shape)
        # print("")        
        
        return x.contiguous(), 1