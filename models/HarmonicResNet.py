import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T

from SURFMNet.utils import SURFMNetLoss, SoftcorNet

# Harmonic Surface Networks components Layers
from nn import HarmonicResNetBlock, ParallelTransportPool, ParallelTransportUnpool, ComplexLin, ComplexNonLin

from transforms import HarmonicPrecomp, ScaleMask, FilterNeighbours

# Utility functions
from utils.harmonic import magnitudes


class HarmonicResNet(torch.nn.Module):
    def __init__(self, nf, max_order, n_rings, scale_transform):
        super(HarmonicResNet, self).__init__()

        self.scale_transform = scale_transform

        self.resnet_block01 = HarmonicResNetBlock(3, nf[0], max_order, n_rings, prev_order=0)

        # Stack 1
        self.resnet_block11 = HarmonicResNetBlock(nf[0], nf[1], max_order, n_rings)
        self.resnet_block12 = HarmonicResNetBlock(nf[1], nf[1], max_order, n_rings)

        # Pooling to scale 1
        self.pool1 = ParallelTransportPool(1, self.scale_transform[1])

        # Stack 2
        self.resnet_block21 = HarmonicResNetBlock(nf[1], nf[2], max_order, n_rings)
        self.resnet_block22 = HarmonicResNetBlock(nf[2], nf[2], max_order, n_rings)

        # Pooling to scale 2
        self.pool2 = ParallelTransportPool(2, self.scale_transform[2])

        # Stack 3
        self.resnet_block31 = HarmonicResNetBlock(nf[2], nf[3], max_order, n_rings)
        self.resnet_block32 = HarmonicResNetBlock(nf[3], nf[3], max_order, n_rings)

        # Pooling to scale 3
        self.pool3 = ParallelTransportPool(3, self.scale_transform[3])

        # Stack 4
        self.resnet_block41 = HarmonicResNetBlock(nf[3], nf[3], max_order, n_rings)
        self.resnet_block42 = HarmonicResNetBlock(nf[3], nf[3], max_order, n_rings)

        # Stack 5
        self.resnet_block51 = HarmonicResNetBlock(nf[3], nf[3], max_order, n_rings)
        self.resnet_block52 = HarmonicResNetBlock(nf[3], nf[3], max_order, n_rings)

        # Unpool to scale 3
        self.unpool3 = ParallelTransportUnpool(3)

        # Stack 6
        self.resnet_block61 = HarmonicResNetBlock(nf[3] + nf[3], nf[2], max_order, n_rings)
        self.resnet_block62 = HarmonicResNetBlock(nf[2], nf[2], max_order, n_rings)

        # Unpool to scale 2
        self.unpool2 = ParallelTransportUnpool(2)

        # Stack 7
        self.resnet_block71 = HarmonicResNetBlock(nf[2] + nf[2], nf[1], max_order, n_rings)
        self.resnet_block72 = HarmonicResNetBlock(nf[1], nf[1], max_order, n_rings)

        # Unpool to scale 1
        self.unpool1 = ParallelTransportUnpool(1)

        # Stack 8
        self.resnet_block81 = HarmonicResNetBlock(nf[1] + nf[1], nf[1], max_order, n_rings)
        self.resnet_block82 = HarmonicResNetBlock(nf[1], nf[1], max_order, n_rings, last_layer=True)

        # Dense final layers
        self.lin1 = ComplexLin(nf[1], 300)
        self.nonlin1 = ComplexNonLin(300, F.rrelu)

    def forward(self, data):
        # We use xyz positions as input in this notebook
        # in the paper, shot descriptors were used
        new_pos = data.pos

        #         x = self.lin0(data.pos, data.edge_index)

        # Linear transformation from input descriptors to nf[0] features
        # Convert input features into complex numbers
        x = torch.stack((data.pos, torch.zeros_like(data.pos)), dim=-1).unsqueeze(1)

        # Stack 1
        # Select only the edges and precomputed components of the first scale
        data_scale0 = self.scale_transform[0](data)

        attributes = (data_scale0.edge_index, data_scale0.precomp, data_scale0.connection)
        x = self.resnet_block01(x, *attributes)
        #         x = self.resnet_block02(x, *attributes)

        x = self.resnet_block11(x, *attributes)
        x_prepool_1 = self.resnet_block12(x, *attributes)

        # Pooling 1
        # Apply parallel transport pooling
        x, data, data_pooled = self.pool1(x_prepool_1, data)

        # Stack 2
        # Store edge_index and precomputed components of the second scale
        attributes_pooled1 = (data_pooled.edge_index, data_pooled.precomp, data_pooled.connection)
        x = self.resnet_block21(x, *attributes_pooled1)
        x_prepool_2 = self.resnet_block22(x, *attributes_pooled1)

        # Pooling 2
        # Apply parallel transport pooling
        x, data, data_pooled = self.pool2(x_prepool_2, data)

        # Stack 3
        # Store edge_index and precomputed components of the third scale
        attributes_pooled2 = (data_pooled.edge_index, data_pooled.precomp, data_pooled.connection)

        x = self.resnet_block31(x, *attributes_pooled2)
        x_prepool_3 = self.resnet_block32(x, *attributes_pooled2)

        # Pooling 3
        # Apply parallel transport pooling
        x, data, data_pooled = self.pool3(x_prepool_3, data)

        # Stack 4
        # Store edge_index and precomputed components of the third scale
        attributes_pooled3 = (data_pooled.edge_index, data_pooled.precomp, data_pooled.connection)
        x = self.resnet_block41(x, *attributes_pooled3)
        x = self.resnet_block42(x, *attributes_pooled3)

        x = self.resnet_block51(x, *attributes_pooled3)
        x = self.resnet_block52(x, *attributes_pooled3)

        data.num_nodes = data.unpool_edges[-1].shape[-1]
        x = self.unpool3(x, data)

        # Concatenate pre-pooling x with post-pooling x
        x = torch.cat((x, x_prepool_3), dim=2)

        # Stack 3
        x = self.resnet_block61(x, *attributes_pooled2)
        x = self.resnet_block62(x, *attributes_pooled2)

        # Unpooling
        data.num_nodes = data.unpool_edges[-1].shape[-1]
        x = self.unpool2(x, data)

        # Concatenate pre-pooling x with post-pooling x
        x = torch.cat((x, x_prepool_2), dim=2)

        # Stack 3
        x = self.resnet_block71(x, *attributes_pooled1)
        x = self.resnet_block72(x, *attributes_pooled1)

        # Unpooling
        data.num_nodes = data.unpool_edges[-1].shape[-1]
        x = self.unpool1(x, data)

        # Concatenate pre-pooling x with post-pooling x
        x = torch.cat((x, x_prepool_1), dim=2)

        # Stack 3
        x = self.resnet_block81(x, *attributes)
        x = self.resnet_block82(x, *attributes)

        x = self.lin1(x)
        x = self.nonlin1(x)

        # Take radial component from features and sum streams
        x = magnitudes(x, keepdim=False)
        x = x.sum(dim=1)

        return x.unsqueeze(0), new_pos.unsqueeze(0)


class SiameseHSN(torch.nn.Module):
    def __init__(self):
        super(SiameseHSN, self).__init__()

        # Maximum rotation order for streams
        max_order = 1

        # Number of rings in the radial profile
        n_rings = 4

        # Number of filters per block
        nf = [8, 16, 32, 48]

        # Radii
        radii = [0.1, 0.2, 0.4, 0.8]

        scale_transform = []


        # Transformations that mask the edges and vertices per scale and precomputes convolution components.
        scale_transform.append(T.Compose((
            ScaleMask(0),
            FilterNeighbours(radius=radii[0]),
            HarmonicPrecomp(n_rings, max_order, max_r=radii[0]))))
        scale_transform.append(T.Compose((
            ScaleMask(1),
            FilterNeighbours(radius=radii[1]),
            HarmonicPrecomp(n_rings, max_order, max_r=radii[1]))))

        scale_transform.append(T.Compose((
            ScaleMask(2),
            FilterNeighbours(radius=radii[2]),
            HarmonicPrecomp(n_rings, max_order, max_r=radii[2]))))

        scale_transform.append(T.Compose((
            ScaleMask(3),
            FilterNeighbours(radius=radii[3]),
            HarmonicPrecomp(n_rings, max_order, max_r=radii[3]))))

        self.net = HarmonicResNet(nf, max_order, n_rings, scale_transform).to("cuda:0")
        self.fmap = SoftcorNet().to("cuda:1")
        self.loss = SURFMNetLoss().to("cuda:2")

    def forward(self, src_data, tar_data):
        src_feat, src_verts = self.net(src_data.to("cuda:0"))
        tar_feat, tar_verts = self.net(tar_data.to("cuda:0"))

        evecs_trans_1 = src_data.evecs_trans[:100, :].unsqueeze(0).to("cuda:1")
        evecs_trans_2 = tar_data.evecs_trans[:100, :].unsqueeze(0).to("cuda:1")

        P12, C12 = self.fmap(src_feat.to("cuda:1"), tar_feat.to("cuda:1"),
                             src_data.evecs[:, :100].unsqueeze(0).to("cuda:1"),
                             tar_data.evecs[:, :100].unsqueeze(0).to("cuda:1"),
                             evecs_trans_1, evecs_trans_2)
        P21, C21 = self.fmap(tar_feat.to("cuda:1"), src_feat.to("cuda:1"),
                             tar_data.evecs[:, :100].unsqueeze(0).to("cuda:1"),
                             src_data.evecs[:, :100].unsqueeze(0).to("cuda:1"),
                             evecs_trans_2, evecs_trans_1)

        evecs_1 = src_data.evecs[:, :100].unsqueeze(0)
        evecs_2 = tar_data.evecs[:, :100].unsqueeze(0)
        evals_1 = src_data.evals[:100].unsqueeze(0).squeeze(-1)
        evals_2 = tar_data.evals[:100].unsqueeze(0).squeeze(-1)

        E1, E2, E3, E4 = self.loss(C12.to("cuda:3"), C21.to("cuda:3"),
                                   src_feat.to("cuda:3"), tar_feat.to("cuda:3"),
                                   evecs_1.to("cuda:3"), evecs_2.to("cuda:3"),
                                   evals_1.to("cuda:3"), evals_2.to("cuda:3"),
                                   torch.device('cuda:3'))

        return C12, C21, src_feat, tar_feat, src_verts, tar_verts, E1, E2, E3, E4, P12, P21
