import torch
import base_trainer
from dataset import OneStyleShapeHF
from torch.utils.data import DataLoader
import networks
import ops
import argparse

device = torch.device("cuda:0")


class HFTrainer(base_trainer.Trainer):
    def load_data(self, split):
        params = self.params
        shape_idx, style_idx = params['shape_style'].split('_')

        dataset = OneStyleShapeHF(self.garment_class, shape_idx=shape_idx, style_idx=style_idx, split=split
                                  , gender=self.gender, smooth_level=params['smooth_level'])
        shuffle = True if split == 'train' else False
        if split == 'train' and len(dataset) > params['batch_size']:
            drop_last = True
        else:
            drop_last = False
        dataloader = DataLoader(dataset, batch_size=self.bs, num_workers=0, shuffle=shuffle,
                                drop_last=drop_last)
        return dataset, dataloader

    def build_model(self):
        params = self.params
        model = getattr(networks, self.net_name)(
            input_size=72, output_size=self.vert_indices.shape[0]*3,
            num_layers=params['num_layers'],
            hidden_size=params['hidden_size']
        )
        return model

    def onestep(self, inputs):
        verts, smooth_verts, thetas, _, _, _ = inputs

        thetas = ops.mask_thetas(thetas, self.garment_class)
        verts = verts.to(device)
        smooth_verts = smooth_verts.to(device)
        thetas = thetas.to(device)

        pred = self.model(thetas).view(verts.shape) + smooth_verts

        loss = (pred - verts).abs().sum(-1).mean()
        return pred, loss

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

    parser.add_argument('--shape_style', default=['000_000'], nargs='+')

    args = parser.parse_args()
    params = args.__dict__

    return params


if __name__ == '__main__':
    params = parse_argument()
    shape_styles = params['shape_style']
    for ss in shape_styles:
        params['shape_style'] = ss
        print("start training, garment:{} shape_style:{}".format(params['garment_class'], ss))
        trainer = HFTrainer(params)

        for i in range(params['start_epoch'], params['max_epoch']):
            print("epoch: {}".format(i))
            trainer.train()

    print("finished")
