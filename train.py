from solver import GANModel
from dataset import get_loader
import torch
import torchvision
import datetime
import yaml
import os
import copy
import time
import argparse

def adjust_learning_rate(optimizer):
    """Learning rate decay scheme"""
    for param_group in optimizer.param_groups:
        param_group['lr'] -= 0.0002/50000
    return param_group['lr']


if __name__ == "__main__":
    # random seed
    torch.manual_seed(0)

    #argument
    parser = argparse.ArgumentParser(
    description='Train a GAN with different yaml files.'
    )
    parser.add_argument('--yaml', type=str, help='Path to yaml file.')
    args = parser.parse_args()

    #load yaml file
    config = yaml.load(open(args.yaml))
    g_batch_size = config['g_batch_size']
    d_batch_size = config['d_batch_size']
    
    #dataset
    dataset = get_loader(image_dir=config['image_dir'], crop_size=config['crop_size'], image_size=config['img_size'],\
         batch_size=d_batch_size, mode='train', dataset_name=config['dataset_name'], num_workers=8)

    #model
    model = GANModel(config)

    num_gens = config['gen_nums']
    ngpu = config['ngpu']
    batch_num = len(dataset)

    #output path
    model_dir = {"checkpoint":"./output/checkpoint", "samples":"./output/samples", "tb":"./output/tensorboard"}
    for dir_ in model_dir:
        if not os.path.exists(model_dir[dir_]):
            os.makedirs(model_dir[dir_])
            
    max_step = config['max_step']
    critic = config['critic']


    #train
    print("start...")
    y = torch.zeros((120, )).long().cuda()
    it = -1
    dataiter = iter(dataset)
    for idx in range(max_step):
        time_start = datetime.datetime.now()
        try:
            data, label = next(dataiter)
        except:
            dataiter = iter(dataset)
            data, label = next(dataiter)

        #update G 
        if idx % critic == 0:
            model.set_input(g_batch_size, data, y)
            D_G_z2 = model.optimize_parametersG()
        #update D
        model.set_input(d_batch_size//num_gens, data, y)
        D_x, D_G_z1 = model.optimize_parametersD()

        #learning rate decay
        if config['lr_decay'] and (idx+1)%config['lr_decay_every'] == 0:
            _ =  adjust_learning_rate(model.optimizer_G)
            lr = adjust_learning_rate(model.optimizer_D)
            model.lr = lr

        #save model
        if (idx+1)%config['backup_every'] == 0:
            #
            torchvision.utils.save_image(model.fake, model_dir['samples'] + '/f{}.jpg'.format(idx+1), nrow=g_batch_size , normalize=True)
            torch.save(model.G.state_dict(), model_dir['checkpoint'] + "/{}_G.pth".format(idx+1))
            torch.save(model.D.state_dict(), model_dir['checkpoint'] +"/{}_D.pth".format(idx+1))
            #
        time_end = datetime.datetime.now()
        print('[%d/%d] D(x): %.4f D(G(z)): %.4f/ %.4f'% (idx+1, max_step, D_x, D_G_z1, D_G_z2))
        #print('alpha: ', model.alpha)
        print("remains {:.4f} minutes...".format((time_end - time_start).total_seconds() / 60. * (max_step - idx)))

            
