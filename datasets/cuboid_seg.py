import os.path as osp
from os import listdir as osls
from os.path import isfile, join
from itertools import combinations
from random import shuffle
from tqdm.notebook import tqdm

import scipy as sp
import scipy.io as sio

import shutil

import torch
from torch_geometric.data import Dataset, extract_zip
from torch_geometric.io import read_off

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import __repr__
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.batch import SimpleBatch
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.utils.enums import ConvolutionFormat
from torch_points3d.utils.config import ConvolutionFormatFactory


class CuboidSeg(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CuboidSeg, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])
        print(len(self.data))
        self.combinations = list(combinations(range(len(self.data)), 2))
        shuffle(self.combinations)

    @property
    def raw_file_names(self):
        return 'cuboid.zip'

    @property
    def processed_file_names(self):
        return ['cuboids.pt']

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        off_path = osp.join(self.raw_dir, 'cuboid_off')
        mat_path = osp.join(self.raw_dir, 'cuboid_mat')

        locations = {f[:-4]: [join(off_path, f), join(mat_path, f[:-3] + 'mat')] for f in
                     osls(off_path) if
                     isfile(join(off_path, f))}

        data_list = []

        for key, value in tqdm(locations.items()):
            data = read_off(value[0])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            mat = sio.loadmat(value[1])
            data.evecs = torch.tensor(mat['evecs'], dtype=torch.float)
            data.evecs_trans = torch.tensor(mat['evecs'].T * sp.sparse.csr_matrix.todense(mat['A']), dtype=torch.float)
            data.evals = torch.tensor(mat['evals'], dtype=torch.float)
            data.name = mat['name']
            data_list.append(data)

        torch.save(data_list, self.processed_paths[0])
        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_off'))
        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_mat'))

    def len(self):
        return len(self.combinations)

    def get(self, idx):
        idx1, idx2 = self.combinations[idx]
        data_src = self.data[idx1]
        data_tar = self.data[idx2]

        return [data_src, data_tar]


class CuboidDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CuboidDataset, self).__init__(root, transform, pre_transform)

        corresp = list(combinations(self.processed_file_names, 2))
        shuffle(corresp)
        self.source_filenames, self.target_filenames = zip(*corresp)

    @property
    def raw_file_names(self):
        return 'cuboid.zip'

    @property
    def processed_file_names(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        off_path = osp.join(self.raw_dir, 'cuboid_off')
        file_names = [f[:-3] + 'pt' for f in osls(off_path) if isfile(osp.join(off_path, f)) and f.endswith('.off')]

        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_off'))
        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_mat'))

        return file_names

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        off_path = osp.join(self.raw_dir, 'cuboid_off')
        mat_path = osp.join(self.raw_dir, 'cuboid_mat')

        locations = {f[:-4]: [join(off_path, f), join(mat_path, f[:-3] + 'mat')] for f in
                     osls(off_path) if
                     isfile(join(off_path, f))}

        for key, value in tqdm(locations.items()):
            data = read_off(value[0])
            print(key)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            mat = sio.loadmat(value[1])
            data.evecs = torch.tensor(mat['evecs'], dtype=torch.float)
            data.evecs_trans = torch.tensor(mat['evecs'].T * sp.sparse.csr_matrix.todense(mat['A']), dtype=torch.float)
            data.evals = torch.tensor(mat['evals'], dtype=torch.float)
            data.name = mat['name']
            fn = key + '.pt'
            torch.save(data, osp.join(self.processed_dir, fn))
        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_off'))
        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_mat'))

    def len(self):
        return len(self.source_filenames)

    def get(self, idx):
        data_src = torch.load(osp.join(self.processed_dir, self.source_filenames[idx]))
        data_tar = torch.load(osp.join(self.processed_dir, self.target_filenames[idx]))

        return [data_src, data_tar]


class KPConvDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(KPConvDataset, self).__init__(root, transform, pre_transform)

        corresp = list(combinations(self.processed_file_names, 2))
        shuffle(corresp)
        self.source_filenames, self.target_filenames = zip(*corresp)

    @property
    def raw_file_names(self):
        return 'cuboid.zip'

    @property
    def processed_file_names(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        off_path = osp.join(self.raw_dir, 'cuboid_off')
        file_names = [f[:-3] + 'pt' for f in osls(off_path) if isfile(osp.join(off_path, f)) and f.endswith('.off')]

        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_off'))
        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_mat'))

        return file_names

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        off_path = osp.join(self.raw_dir, 'cuboid_off')
        mat_path = osp.join(self.raw_dir, 'cuboid_mat')

        locations = {f[:-4]: [join(off_path, f), join(mat_path, f[:-3] + 'mat')] for f in
                     osls(off_path) if
                     isfile(join(off_path, f))}

        for key, value in tqdm(locations.items()):
            data = read_off(value[0])
            print(key)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            mat = sio.loadmat(value[1])
            data.evecs = torch.tensor(mat['evecs'], dtype=torch.float)
            data.evecs_trans = torch.tensor(mat['evecs'].T * sp.sparse.csr_matrix.todense(mat['A']), dtype=torch.float)
            data.evals = torch.tensor(mat['evals'], dtype=torch.float)
            data.name = mat['name']
            fn = key + '.pt'
            torch.save(data, osp.join(self.processed_dir, fn))
        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_off'))
        shutil.rmtree(osp.join(self.raw_dir, 'cuboid_mat'))

    def len(self):
        return len(self.source_filenames)

    def get(self, idx):
        data_src = torch.load(osp.join(self.processed_dir, self.source_filenames[idx]))
        data_tar = torch.load(osp.join(self.processed_dir, self.target_filenames[idx]))

        return [data_src, data_tar]
