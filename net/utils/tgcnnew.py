# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
from torch.distributions.utils import broadcast_all 

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
        
            # nn.Dropout(0.1, inplace=True),
        self.pre = nn.Conv2d(out_channels * kernel_size, out_channels,kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        
    # def forward(self, x, A):
    #     assert A.size(0) == self.kernel_size

    #     x = self.conv(x)

    #     n, kc, t, v = x.size()
    #     x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
    #     x = torch.einsum('nkctv,kvw->nctw', (x, A))

    #     return x.contiguous(), A

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        # print("")

        x = self.conv(x)

        # print(x.shape)

        n, kc, t, v = x.size()
        # result = x
        x = x.view(n, self.kernel_size, (kc//self.kernel_size) * t, v)
        # print("")

        # print("")
        # print(A.unsqueeze(1).unsqueeze(0).shape)
        # print(x.unsqueeze(-1).shape)
        # print(A.unsqueeze(1).unsqueeze(1).unsqueeze(0).shape)
        # print(torch.sum(torch.mul(x.unsqueeze(-1), A.unsqueeze(1).unsqueeze(1).unsqueeze(0)), (1, 4)))
        # print(torch.einsum('nkctv,kvw->nctw', (x, A)))
        # A = A.unsqueeze(0)
        # # q, _, _, _ = A.size()
        # print( n, kc, t, v)
        # print(x.shape)
        #13 3 x 25
        #1  3 x 25
        # print(A.shape)
        #1 3 25 25
        # x = torch.einsum('nctv,acvf->nctv', x, A)
        # x = torch.sum(x, 1)
        # print(x.shape)
        # x = x.view(n, kc//self.kernel_size, t, v)
        # print(x.shape)
        
        # print("")
        # print(result.shape)
        
        # result = result.view(n * self.kernel_size, (kc // self.kernel_size) * t, v)
        # result = torch.matmul(result, A.view(kc, -1)).view(n, -1, w)
        # result = result.view(n, kc // kernel_size, t, v)
        # print(result.shape)
        
        

        # x = torch.einsum('nkcv,kvw->ncw', (x, A))
        # x = torch.sum(torch.mul(x, A.unsqueeze(1).unsqueeze(0)), (1, 3))

        
 
        x_splits = torch.split(x, 1, 1)
        A_splits = torch.split(A.unsqueeze(0), 1, 1)
        

        # print("before mul opp shape :")
        # print(x_splits[0].view(n, (kc//self.kernel_size) * t, v, 1).shape)
        # print(A_splits[0].shape)
        
        # x0 = x_splits[0].view(n, (kc//self.kernel_size)*t, v, 1)
        # x1 = x_splits[1].view(n, (kc//self.kernel_size)*t, v, 1)
        # x2 = x_splits[2].view(n, (kc//self.kernel_size)*t, v, 1)
        
        
        # s1, s2, s3, s4 = x0.shape
        # a1, a2, a3, a4 = A_splits[0].shape
   
        # desired_shape = (s1, s2, s3, a4)
        
        # x0_broad = torch.empty(desired_shape)
        # x1_broad = torch.empty(desired_shape)
        # x2_broad = torch.empty(desired_shape)
        # a0_broad = torch.empty(desired_shape)
        # a1_broad = torch.empty(desired_shape)
        # a2_broad = torch.empty(desired_shape)
                
        # for i in range(desired_shape[0]):
        #     for j in range(desired_shape[1]):
        #         for k in range(desired_shape[2]):
        #             x0_broad[i, j, k, :] = x0[0, 0, k, :]
        #             x1_broad[i, j, k, :] = x1[0, 0, k, :]
        #             x2_broad[i, j, k, :] = x2[0, 0, k, :]
        # print(x0_broad.shape)
                
        # for i in range(desired_shape[0]):
        #     for j in range(desired_shape[1]):
        #         for k in range(desired_shape[2]):
        #             a0_broad[i, j, k, :] = A_splits[0][0, 0, k, :]        
        #             a1_broad[i, j, k, :] = A_splits[1][0, 0, k, :]        
        #             a2_broad[i, j, k, :] = A_splits[2][0, 0, k, :]        
        # print(a0_broad.shape)
        
        # b0 = torch.sum(torch.mul(x0_broad, a0_broad), 2)
        # b1 = torch.sum(torch.mul(x1_broad, a1_broad), 2)
        # b2 = torch.sum(torch.mul(x2_broad, a2_broad), 2)
        
        # max1, max2, max3, max4 = max(s1, a1), max(s2, a2), max(s3, a3), max(s4, a4)
        
        # x0_reshape = x0.expand(max1, max2, max3, max4)
        # x1_reshape = x1.expand(max1, max2, max3, max4)
        # x2_reshape = x2.expand(max1, max2, max3, max4)
        
        # a0_reshape = A_splits[0].expand(max1, max2, max3, max4)
        # a1_reshape = A_splits[1].expand(max1, max2, max3, max4)
        # a2_reshape = A_splits[2].expand(max1, max2, max3, max4)
        
        # print("x0 shape : ", x0_reshape.shape)
        # print("a0 shape : ", a0_reshape.shape)
        
        # b0 = torch.sum(torch.mul(x0_reshape, a0_reshape), 2)
        # b1 = torch.sum(torch.mul(x1_reshape, a1_reshape), 2)
        # b2 = torch.sum(torch.mul(x2_reshape, a2_reshape), 2)
        
        # -[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

        # b0 = torch.sum(torch.mul(broadcast_all(x0, A_splits[0])[0], broadcast_all(x0, A_splits[0])[1]), 2)
        # b1 = torch.sum(torch.mul(broadcast_all(x0, A_splits[0])[0], broadcast_all(x0, A_splits[0])[1]), 2)
        # b2 = torch.sum(torch.mul(broadcast_all(x0, A_splits[0])[0], broadcast_all(x0, A_splits[0])[1]), 2)
        # x0_re = x0
        # x1_re = x1
        # x2_re = x2
        
        # a0_re = A_splits[0]
        # a1_re = A_splits[1]
        # a2_re = A_splits[2]
        
        # for i in range(0, a4-1):
        #     x0_re = torch.concat((x0_re, x0), 3)



        #     x1_re = torch.concat((x1_re, x1), 3)
        #     x2_re = torch.concat((x2_re, x2), 3)

        # for i in range(0, s2-1):
        #     a0_re = torch.concat((a0_re, A_splits[0]), 1)
        #     a1_re = torch.concat((a1_re, A_splits[1]), 1)
        #     a2_re = torch.concat((a2_re, A_splits[2]), 1)


            
            
        # b0 = torch.sum(torch.mul(x0_re, a0_re), 2)
        # b1 = torch.sum(torch.mul(x1_re, a1_re), 2)
        # b2 = torch.sum(torch.mul(x2_re, a2_re), 2)
        
        # b0 = torch.sum(torch.mul(x0.repeat(1, 1, 1, a4), A_splits[0].repeat(s1, s2, 1, 1)), 2)
        # b1 = torch.sum(torch.mul(x1.repeat(1, 1, 1, a4), A_splits[1].repeat(s1, s2, 1, 1)), 2)
        # b2 = torch.sum(torch.mul(x2.repeat(1, 1, 1, a4), A_splits[2].repeat(s1, s2, 1, 1)), 2)
        
        # [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
        b0 = torch.sum(torch.mul(x_splits[0].view(n, (kc//self.kernel_size) * t, v, 1), A_splits[0]), 2)
        b1 = torch.sum(torch.mul(x_splits[1].view(n, (kc//self.kernel_size) * t, v, 1), A_splits[1]), 2)
        b2 = torch.sum(torch.mul(x_splits[2].view(n, (kc//self.kernel_size) * t, v, 1), A_splits[2]), 2)
        
        # print("after mul opp shape :")
        # print(b0.shape)
        
        x = torch.stack((b0, b1, b2), 1)
        x = torch.sum(x, 1)
        # print(x.shape)
        
        x = x.view(n, kc//self.kernel_size, t, v)
        # x = self.pre(x)
        
        return x.contiguous(), 1