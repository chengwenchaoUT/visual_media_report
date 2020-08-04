from torch.utils.data import DataLoader
import torch
import base_trainer
from dataset import ShapeStyleInCannonPose
import networks
import argparse

device = torch.device("cuda:0")


class CannonTrainer(base_trainer.Trainer):
    def load_data(self, split):
        params = self.params
        dataset = ShapeStyleInCannonPose(self.garment_class, split=split, gender=self.gender)
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
            input_size=10+4, output_size=self.vert_indices.shape[0] * 3,
            num_layers=params['num_layers'],
            hidden_size=params['hidden_size']
        )
        return model

    def onestep(self, inputs):
        verts, _, betas, gammas, _ =inputs

        verts = verts.to(device)
        betas = betas.to(device)
        gammas = gammas.to(device)
        pred = self.model(
            torch.cat((betas, gammas), dim=1)).view(verts.shape)

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
    print("start training CannonSS {}".format(params['garment_class']))
    trainer = CannonTrainer(params)

    for i in range(params['start_epoch'], params['max_epoch']):
        print("epoch: {}".format(i))
        trainer.train()

    print("finished")


