"""Output a .npy file"""

from solver import GANModel
import torch
import torchvision
from PIL import Image
import yaml
import numpy as np
import argparse


def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]


if __name__ == "__main__":

    #argument
    parser = argparse.ArgumentParser(
    description='Test a GAN with different config.'
    )
    parser.add_argument('--yaml', type=str, help='Path to yaml file.')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpont.')
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--output_name', type=str, help='Name to saved .npy file')
    args = parser.parse_args()
    
    config = yaml.load(open(args.yaml))
    gen_nums = config['gen_nums']
    z_dim = config['z_dim']
    g_batch_size = config['g_batch_size']
    

    model = GANModel(config)
    model.G.load_state_dict(torch.load(args.checkpoint))
    model.G.eval()

    assert args.num_samples % gen_nums == 0
    z_nums = args.num_samples // gen_nums

    noise = np.random.randn(z_nums, z_dim).astype(np.float32)
    batches = make_batches(z_nums, g_batch_size)

    h = []
    y = torch.zeros((120, )).long().cuda()
    for batch_idx, (batch_start, batch_end) in enumerate(batches):
        noise_batch = noise[batch_start:batch_end]
        noise_batch = torch.from_numpy(noise_batch).cuda()
        out = model.G(noise_batch, y).detach().cpu().numpy()
        out = np.multiply(np.add(np.multiply(out, 0.5), 0.5), 255).astype('int32')
        #out = out[sli]
        h.append(out)
    h = np.vstack(h)
    h = np.transpose(h, (0, 2, 3, 1))
    np.save(args.output_name, h)
