from torch.utils.data import Dataset, ConcatDataset
import os
import local_config
import numpy as np
import torch
import pickle
from smoothing import DiffusionSmoothing
from SMPLToGarment import TorchSMPLToGarment

def flip_theta(theta, batch=False):
    """
    flip SMPL theta along y-z plane
    if batch is True, theta shape is Nx72, otherwise 72
    """
    exg_idx = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    if batch:
        new_theta = np.reshape(theta, [-1, 24, 3])
        new_theta = new_theta[:, exg_idx]
        new_theta[:, :, 1:3] *= -1
    else:
        new_theta = np.reshape(theta, [24, 3])
        new_theta = new_theta[exg_idx]
        new_theta[:, 1:3] *= -1
    new_theta = new_theta.reshape(theta.shape)
    return new_theta

def get_Apose():
    with open(os.path.join(local_config.DATA_DIR, 'apose.pkl'), 'rb') as f:
        apose = np.array(pickle.load(f, encoding='latin1')['pose']).astype(np.float32)

    flip_pose = flip_theta(apose)
    apose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]] = 0
    apose[[14, 17, 19, 21, 23]] = flip_pose[[14, 17, 19, 21, 23]]
    apose = apose.reshape([72])
    return apose

# Smoothing levels can be defined here.
# smooth level 0 is not smoothing.
# smooth level 1 is smoothing with 0.15 smoothness for 80 iterations.
level_smoothness = [0, 0.15]
level_smoothiter = [0, 80]
Ltype = "uniform"

def smooth_it(smoothing, smooth_level, smpl, thetas, betas, verts, garment_class):
    """Smoothing function used only when smoothing is done during training time."""
    if smooth_level == -1:
        verts = torch.zeros_like(verts)
    elif smooth_level != 0:
        v_poseshaped = smpl.forward_poseshaped(
            theta=thetas.unsqueeze(0), beta=betas.unsqueeze(0),
            garment_class=garment_class)[0]
        unposed_gar_smooth = (v_poseshaped + verts).numpy()
        unposed_gar_smooth = smoothing.smooth(
            unposed_gar_smooth, smoothness=level_smoothness[smooth_level],
            Ltype=Ltype, n=level_smoothiter[smooth_level])
        verts = torch.from_numpy(unposed_gar_smooth.astype(np.float32)) - v_poseshaped
    return verts

class PivotsStyleShape(Dataset):
    def __init__(self, garment_class, split, gender, smooth_level, smpl=None):
        super(PivotsStyleShape, self).__init__()

        self.garment_class = garment_class
        self.smooth_level = smooth_level
        self.split = split
        self.gender = gender
        self.smpl = smpl
        assert (gender in ['neutral', 'male', 'female'])
        assert (split in ['train', 'test', None, 'train_train',
                          'train_test', 'test_train', 'test_test'])

        self.datasets = self.get_datasets()
        self.ds = ConcatDataset(self.datasets)
        if smooth_level == 1 and local_config.SMOOTH_STORED:
            print("Using Smoothing in the dataset")
            return
        if self.smooth_level != 0 and self.smooth_level != -1:
            print("Using Smoothing in the dataset")
            print(self.smooth_level, Ltype)
            with open(os.path.join(local_config.DATA_DIR, local_config.GAR_INFO_FILE), 'rb') as f:
                class_info = pickle.load(f)
            num_v = len(class_info[garment_class]['vert_indices'])
            self.smoothing = DiffusionSmoothing(
                np.zeros((num_v, 3)), class_info[garment_class]['f'])
            self.smpl = TorchSMPLToGarment(gender=gender)
        else:
            self.smoothing = None
            self.smpl = None


    def get_datasets(self):
        garment_class, split, gender = self.garment_class, self.split, self.gender
        data_dir = os.path.join(local_config.DATA_DIR, '{}_{}'.format(garment_class, gender))
        with open(os.path.join(data_dir, "pivots.txt"), 'r') as f:
            train_pivots = [l.strip().split('_') for l in f.readlines()]

        test_path = os.path.join(data_dir, 'test.txt')
        if os.path.exists(test_path):
            with open(test_path, 'r') as f:
                test_pivots = [l.strip().split('_') for l in f.readlines()]
        else:
            print("cannot find test pivots file.")
            test_pivots =[]

        sl = 0
        if self.smooth_level == 1 and local_config.SMOOTH_STORED:
            sl = 1

        datasets = []

        for shape_index, style_index in train_pivots:
            datasets.append(
                OneStyleShape(garment_class, shape_idx=shape_index,
                              style_idx=style_index, split=split,
                              gender=gender, smooth_level=sl))

        for shape_index, style_index in test_pivots:
            if split == 'train': continue
            datasets.append(
                OneStyleShape(garment_class, shape_idx=shape_index,
                              style_idx=style_index, split=None,
                              gender=gender, smooth_level=sl))

        return datasets

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        verts, thetas, betas, gammas, _ = self.ds[item]
        if self.smooth_level == 1 and local_config.SMOOTH_STORED:
            return verts, thetas, betas, gammas, item
        verts = smooth_it(self.smoothing, self.smooth_level,
                          self.smpl, thetas, betas, verts, self.garment_class)
        return verts, thetas, betas, gammas, item


class OneStyleShape(Dataset):
    def __init__(self, garment_class, shape_idx, style_idx, split, gender, smooth_level):
        super(OneStyleShape, self).__init__()

        self.garment_class = garment_class
        self.split = split
        self.gender = gender
        self.shape_idx = shape_idx
        self.style_idx = style_idx
        self.smooth_level = smooth_level

        data_dir = os.path.join(local_config.DATA_DIR, '{}_{}'.format(garment_class, gender))

        beta = np.load(os.path.join(data_dir, 'shape/beta_{}.npy'.format(shape_idx)))
        gamma = np.load(os.path.join(data_dir, 'style/gamma_{}.npy'.format(style_idx)))

        thetas = []
        pose_order = []
        verts_d = []
        smooth_verts_d = []
        pose_idx = 0
        while True:
            pose_path = os.path.join(data_dir, 'pose/{}_{}/poses_{:03d}.npz'.format(shape_idx, style_idx, pose_idx))
            if not os.path.exists(pose_path):
                break
            pose_data = np.load(pose_path)
            verts_d_path = os.path.join(data_dir, 'pose/{}_{}/unposed_{:03d}.npy'.format(shape_idx, style_idx, pose_idx))
            if not os.path.exists(verts_d_path):
                print("{} doesn't not exists".format(verts_d_path))
                pose_idx += 1
                continue

            thetas.append(pose_data['thetas'])
            pose_order.append(pose_data['pose_order'])
            verts_d.append(np.load(verts_d_path))

            if smooth_level == 1 and local_config.SMOOTH_STORED:
                smooth_verts_d_path = os.path.join(data_dir, 'pose/{}_{}/unposed_{:03d}.npy'.format(shape_idx, style_idx, pose_idx))
                if not os.path.exists(smooth_verts_d_path):
                    print("{} doesn't exist.".format(smooth_verts_d_path))
                    exit(-1)
                smooth_verts_d.append(np.load(smooth_verts_d_path))

            pose_idx += 1

        thetas = np.concatenate(thetas, axis=0)
        pose_order = np.concatenate(pose_order, axis=0)
        verts_d = np.concatenate(verts_d, axis=0)
        if smooth_level == 1 and local_config.SMOOTH_STORED:
            smooth_verts_d = np.concatenate(smooth_verts_d, axis=0)

        if split is not None:
            assert(split in ['test', 'train'])
            split_path = os.path.join(local_config.DATA_DIR, local_config.POSE_SPLIT_FILE)
            if pose_idx>1:
                test_s_idx = np.load(split_path)['test']
                mask = np.in1d(pose_order, test_s_idx)
                mask_idx = np.where(mask)[0] if split == 'test' else np.where(~mask)[0]
            else:
                train_s_idx = np.load(split_path)['train']
                mask = np.in1d(pose_order, train_s_idx)
                mask_idx = np.where(mask)[0] if split == 'train' else np.where(~mask)[0]

            thetas = thetas[mask_idx]
            verts_d = verts_d[mask_idx]
            if smooth_level == 1 and local_config.SMOOTH_STORED:
                smooth_verts_d = smooth_verts_d[mask_idx]

        self.verts_d = torch.from_numpy(verts_d.astype(np.float32))
        self.thetas = torch.from_numpy(thetas.astype(np.float32))
        self.beta = torch.from_numpy(beta[:10].astype(np.float32))
        self.gamma = torch.from_numpy(gamma.astype(np.float32))
        if smooth_level == 1 and local_config.SMOOTH_STORED:
            self.smooth_verts_d = torch.from_numpy(smooth_verts_d.astype(np.float32))
            return

        if self.smooth_level != 0 and self.smooth_level != -1:
            with open(os.path.join(local_config.DATA_DIR, local_config.GAR_INFO_FILE), 'rb') as f:
                class_info = pickle.load(f)
            num_v = len(class_info[garment_class]['vert_indices'])
            self.smoothing = DiffusionSmoothing(np.zeros((num_v, 3)), class_info[garment_class]['f'])
            self.smpl = TorchSMPLToGarment(gender=gender)
        else:
            self.smoothing = None
            self.smpl = None

    def __len__(self):
        return self.thetas.shape[0]

    def __getitem__(self, item):
        verts_d, theta, beta, gamma = self.verts_d[item], self.thetas[item], self.beta, self.gamma
        if self.smooth_level == 1 and local_config.SMOOTH_STORED:
            verts_d = self.smooth_verts_d[item]
        else:
            verts_d = smooth_it(self.smoothing, self.smooth_level, self.smpl,
                                theta, beta, verts_d, self.garment_class)
        return verts_d, theta, beta, gamma, item


class OneStyleShapeHF(OneStyleShape):
    def __init__(self, garment_class, shape_idx, style_idx, split, gender, smooth_level):
        super(OneStyleShapeHF, self).__init__(garment_class, shape_idx, style_idx, split, gender, smooth_level)

    def __getitem__(self, item):
        verts_d = self.verts_d[item]
        ret = super(OneStyleShapeHF, self).__getitem__(item)
        hf = (verts_d,) + ret
        return hf


class ShapeStyleInCannonPose(Dataset):
    def __init__(self, garment_class, gender, ss_path='avail.txt', split=None):
        super(ShapeStyleInCannonPose, self).__init__()
        self.garment_class = garment_class
        self.gender = gender
        data_dir = os.path.join(local_config.DATA_DIR, '{}_{}'.format(garment_class, gender))

        betas = np.stack([np.load(os.path.join(data_dir, 'shape/beta_{:03d}.npy'.format(i))) for i in range(9)]).astype(np.float32)[:, :10]
        gammas = np.stack([np.load(os.path.join(data_dir, 'style/gamma_{:03d}.npy'.format(i))) for i in range(26)]).astype(np.float32)

        with open(os.path.join(data_dir, ss_path), 'r') as f:
            ss = [l.strip().split('_') for l in f.readlines()]

        assert(split in [None, 'train', 'test'])
        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            test_ss = [l.strip().split('_') for l in f.readlines()]
        if split == 'train':
            ss = [item for item in ss if item not in test_ss]
        elif split == 'test':
            ss = [item for item in ss if item in test_ss]

        unpose_vert = []
        for shape_idx, style_idx in ss:
            path = os.path.join(data_dir, 'style_shape/beta{}_gamma{}.npy'.format(shape_idx, style_idx))
            unpose_vert.append(np.load(path))
        unpose_vert = np.stack(unpose_vert)

        self.ss = ss
        self.betas = torch.from_numpy(betas.astype(np.float32))
        self.gammas = torch.from_numpy(gammas.astype(np.float32))
        self.unpose_v = torch.from_numpy(unpose_vert.astype(np.float32))
        self.apose = torch.from_numpy(get_Apose().astype(np.float32))

    def __len__(self):
        return self.unpose_v.shape[0]

    def __getitem__(self, item):
        shape, style = self.ss[item]
        shape, style = int(shape), int(style)
        return self.unpose_v[item], self.apose, self.betas[shape], self.gammas[style], item
