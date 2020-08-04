import torch
from SMPLToGarment import SMPLToGarment
import local_config
import os
import pickle
import numpy as np
from dataset import PivotsStyleShape
from torch.utils.data import DataLoader
import networks
import ops
import argparse

device = torch.device("cuda:0")

class Trainer(object):
    """
    trainer class for TailorNet baseline
    """
    def __init__(self, params):
        self.params = params
        self.gender = params['gender']
        self.garment_class = params['garment_class']

        self.bs = params['batch_size']
        self.net_name = params['net_name']

        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        # smpl
        self.smpl = SMPLToGarment(gender=self.gender)

        # garment
        with open(os.path.join(local_config.DATA_DIR, local_config.GAR_INFO_FILE), 'rb') as f:
            class_info = pickle.load(f)
        self.body_f_np = self.smpl.smpl.f.astype(np.long)
        self.garment_f_np = class_info[self.garment_class]['f']
        self.garment_f_tensor = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)
        self.vert_indices = class_info[self.garment_class]['vert_indices']

        # dataset
        self.train_dataset, self.train_dataloader = self.load_data('train')
        self.test_dataset, self.test_dataloader = self.load_data('test')

        # networks
        self.model = self.build_model()
        self.model.to(device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
        )

    def load_data(self, split):
        params = self.params
        dataset = PivotsStyleShape(self.garment_class, split=split, gender=self.gender, smooth_level=params['smooth_level'])
        shuffle = True if split == 'train' else False
        if split == 'train' and len(dataset) > params['batch_size']:
            drop_last = True
        else:
            drop_last = False
        dataloader = DataLoader(dataset, batch_size=self.bs, num_workers=0, shuffle=shuffle, drop_last=drop_last)
        return dataset, dataloader

    def build_model(self):
        params = self.params
        model = getattr(networks, self.net_name)(
            input_size=72+10+4, output_size=self.vert_indices.shape[0]*3,
            num_layers=params['num_layers'],
            hidden_size=params['hidden_size']
        )
        return model

    def train(self):
        self.model.train()
        for i, inputs in enumerate(self.train_dataloader):
            self.optim.zero_grad()
            _, loss = self.onestep(inputs)
            loss.backward()
            self.optim.step()

            print("Iter {}, loss: {:.8f}".format(self.iter_nums, loss.item()))
            self.iter_nums += 1

    def onestep(self, inputs):
        verts, thetas, betas, gammas, _ = inputs

        thetas = ops.mask_thetas(thetas, self.garment_class)

        verts = verts.to(device)
        thetas = thetas.to(device)
        betas = betas.to(device)
        gammas = gammas.to(device)
        pred_verts = self.model(torch.cat((thetas, betas, gammas), dim=1)).view(verts.shape)

        loss = (pred_verts - verts).abs().sum(-1).mean()
        return pred_verts, loss

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--garment_class', default="old-t-shirt")
    parser.add_argument('--gender', default="female")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--smooth_level', default=0, type=int)
    parser.add_argument('--net_name', default="FullyConnected")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--hidden_size', default=1048)

    args = parser.parse_args()
    params = args.__dict__

    return params

if __name__ == '__main__':
    params = parse_argument()

    print("start training {}".format(params['garment_class']))
    trainer = Trainer(params)

    for i in range(params['start_epoch'], params['max_epoch']):
        print("epoch: {}".format(i))
        trainer.train()

    print("finished!")

