import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py

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
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A


class seq_gcn(nn.Module):
    # Source: https://github.com/abduallahmohamed/Social-STGCNN/blob/master/model.py
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
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
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(seq_gcn, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        return x, A


class label_gcnn(nn.Module):
    def __init__(self, n_layer=1,  input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=2, kernel_size=3, hot_enc_length=1, class_enc=True):
        super(label_gcnn, self).__init__()

        self.class_enc = class_enc
        if class_enc:
            self.v_norm = nn.Sequential(
                nn.Linear(in_features=seq_len, out_features=seq_len),
                nn.PReLU(),
            )
            self.a_norm = nn.Sequential(
                nn.Linear(in_features=seq_len, out_features=seq_len),
                nn.PReLU(),
            )

            self.a_lin1 = nn.Sequential(
                nn.Linear(in_features=2 * hot_enc_length, out_features=seq_len),
                nn.PReLU(),
            )
            self.a_lin2 = nn.Sequential(
                nn.Linear(in_features=2 * seq_len, out_features=seq_len),
                nn.PReLU(),

            )

        self.n_layer = n_layer

        self.seq_gcns = nn.ModuleList()
        self.seq_gcns.append(seq_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_layer):
            self.seq_gcns.append(seq_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        #self.tpcnns = nn.ModuleList()
        #self.tpcnns.append(nn.Conv2d(seq_len, 1, 3, padding=1))
        self.pred_embed = nn.Sequential(nn.Linear(seq_len, pred_seq_len, bias=True), nn.PReLU()) # predict 1 step further


    def forward(self, v, a, hot_enc): 
        # pedestrians that are within 1 pixel have same similarity as person they are next to
        # a = torch.where(a > 1, torch.ones_like(a), a)
        if self.class_enc:
            #normalise inputs with layers
            v = self.v_norm(v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            a = self.a_norm(a.permute(1, 2, 0)).permute(2, 0, 1)
            #combine class labels with adjacency matrix
            hot_enc = hot_enc.repeat(a.shape[1], 1, 1)
            hot_enc = torch.cat((hot_enc.rot90(k=-1), hot_enc), 2)

            #linear
            c = self.a_lin1(hot_enc).permute(2, 0, 1)# 8 31 31 #.squeeze().repeat(a.shape[0], 1, 1)
            a = self.a_lin2(torch.cat((a, c)).permute(1, 2, 0)).permute(2, 0, 1) # 8 31 31

        for k in range(self.n_layer):
            v, a = self.seq_gcns[k](v, a)

        v = v.permute(0, 1, 3, 2)
        v = self.pred_embed(v)
        v = v.permute(0, 1, 3, 2)


        return v, a

