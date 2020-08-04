from SmplPaths import SmplPaths
from smpl_lib.ch_smpl import Smpl
import os
import pickle
import torch
import torch.nn as nn
import local_config
import numpy as np

class SMPLToGarment(object):
    """
    SMPL class
    """
    def __init__(self, gender):
        self.gender = gender
        smpl_model = SmplPaths(gender=gender).get_smpl_data()

        self.smpl = Smpl(smpl_model)
        with open(os.path.join(local_config.DATA_DIR, local_config.GAR_INFO_FILE), 'rb') as f:
            self.class_info = pickle.load(f)


class TorchSMPLToGarment(nn.Module):
    """Pytorch version of SMPLToGarment class."""
    def __init__(self, gender):
        super(TorchSMPLToGarment, self).__init__()

        model = SmplPaths(gender=gender).get_smpl_data()
        with open(os.path.join(local_config.DATA_DIR, local_config.GAR_INFO_FILE), 'rb') as f:
            class_info = pickle.load(f)
        for k in class_info.keys():
            if isinstance(class_info[k]['vert_indices'], np.ndarray):
                class_info[k]['vert_indices'] = torch.tensor(
                    class_info[k]['vert_indices'].astype(np.int64))
            if isinstance(class_info[k]['f'], np.ndarray):
                class_info[k]['f'] = torch.tensor(class_info[k]['f'].astype(np.int64))

        self.class_info = class_info
        self.gender = gender

        self.faces = model['f']

        np_v_template = np.array(model['v_template'], dtype=np.float)

        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)[:, :, :10]
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = np.array(model['J_regressor'].todense(), dtype=np.float).T
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_joint_regressor = np.array(model['J_regressor'].todense(), dtype=np.float)
        self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        np_weights = np.array(model['weights'], dtype=np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        self.register_buffer(
            'weight',
            torch.from_numpy(np_weights).float().reshape(1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None
        self.num_verts = 27554

    def forward_poseshaped(self, theta, beta=None, garment_class=None):
        if not self.cur_device:
            device = theta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = theta.shape[0]

        if beta is not None:
            v_shaped = torch.matmul(
                beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_shaped = self.v_template.unsqueeze(0).expand(num_batch, -1, -1)
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(
            pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        if garment_class is not None:
            v_posed = v_posed[:, self.class_info[garment_class]['vert_indices']]
        return v_posed


def batch_rodrigues(theta):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)