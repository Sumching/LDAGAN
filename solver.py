"""Define a solver"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.special as f
from network import defineNet
from torch import autograd
import copy
import numpy as np
from network import defineNet
from utils.tools import alpha_Adam, kl_loss, compute_grad2


class GANModel():
    def __init__(self, config):
        self.ngpu = config['ngpu']
        self.batch_size = config['g_batch_size']
        self.d_batch_size = config['d_batch_size']
        self.num = config['gen_nums']
        self.lr = config['lr']
        self.device = 'cuda:0'
        self.z_dim = config['z_dim']
        self.max_step = config['max_step']
        self.network_name = config['network_name']
        self.info_bottle = config['info_bottle']        
        Generator, Discriminator = defineNet(config['img_size'])
        self.G = Generator(num_gens=self.num).to(self.device)
        self.D = Discriminator().to(self.device)
        if ('cuda' in self.device) and (self.ngpu > 1):
            self.G = nn.DataParallel(self.G, list(range(self.ngpu)))
            self.D = nn.DataParallel(self.D, list(range(self.ngpu)))
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))


        if self.network_name == "vgan-based":
            # Average model
            self.G_test = copy.deepcopy(self.G)
            self.reg_param = 0.1
            self.target_kl= 0.2
            self.beta_step = 0.00001
            self.z_fixed = torch.randn(self.batch_size*self.num, self.z_dim).to(self.device)
            self.optimizer_G = torch.optim.RMSprop(self.G.parameters(), lr=self.lr, alpha=0.99, eps=1e-8)
            self.optimizer_D = torch.optim.RMSprop(self.D.parameters(), lr=self.lr, alpha=0.99, eps=1e-8)

        self.al_optim = alpha_Adam(beta1=0., beta2 = .9,)
        #initialize the Dirichlet parameter alpha
        alpha_init = config['alpha_init']
        alpha = np.ones([1, self.num])
        self.alpha = alpha * alpha_init
        alpha_t = np.concatenate(([alpha for i in range(self.batch_size)]), 0)
        
        #initialize the parameter w and gamma
        self.w = np.ones([self.batch_size, self.num]) / (self.num)
        self.gamma = self.w + alpha_t


    def set_input(self, imgs_eachgen, data, y):
        """Prepare data to be sent to the network"""
        if self.network_name == "vgan-based":
            imgs_eachgen = imgs_eachgen * self.num
        z = torch.randn(imgs_eachgen, self.z_dim)
        self.z = z.to(self.device)
        self.real = data.to(self.device)
        self.y = y.to(self.device)


    def set_requires_grad(self, nets, requires_grad=False):
        """Start or stop the gradient calculating"""
        for param in nets.parameters():
            param.requires_grad_(requires_grad)

    def forward(self):
        """Forward process"""
        fake = self.G(self.z, self.y)

        #change images order, like G1 G2...G10, when gpu >= 2
        bs, c, h, w = fake.shape
        fake = fake.view(bs//self.ngpu, self.ngpu, c, h, w)
        fakes = fake.chunk(self.ngpu, 1)
        self.fake = torch.cat(fakes, dim=0).view(bs, c, h, w)


        

    def backward_D(self):
        """Calculate the gradient of discriminator"""
        self.G.train()
        self.D.train()
        if self.network_name == 'sngan-based':
            x_real = self.real
            output = self.D(x_real, self.y).view(-1)
            #hine loss
            errD_real = torch.mean(F.relu(1. - output))
            errD_real.backward()
            D_x = output.mean().item()
            output = self.D(self.fake.detach(), self.y)
            errD_fake = torch.mean(F.relu(1. + output.view(-1)))
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = errD_real + errD_fake
        else:
            #vgan-based

            #on real data
            x_real = self.real
            x_real.requires_grad_()

            d_real_dict = self.D(x_real, self.y)
            d_real = d_real_dict['out']
            dloss_real = self.compute_loss(d_real, 1)
            dloss_real.backward(retain_graph=True)

            reg = 0.

            
            reg += 10. * compute_grad2(d_real, x_real).mean()#gradinent penalty

            if self.info_bottle:
                mu = d_real_dict['mu']
                logstd = d_real_dict['logstd']
                kl_real = kl_loss(mu, logstd).mean()


            d_acc_real = torch.mean((d_real > 0.5).float())

            # On fake data
            x_fake = self.fake
            x_fake.requires_grad_()
            d_fake_dict = self.D(x_fake, self.y)
            d_fake = d_fake_dict['out']
            dloss_fake = self.compute_loss(d_fake, 0)

            if self.info_bottle:
                dloss_fake.backward(retain_graph=True)
                mu_fake = d_fake_dict['mu']
                logstd_fake = d_fake_dict['logstd']
                kl_fake = kl_loss(mu_fake, logstd_fake).mean()
                avg_kl = 0.5 * (kl_real + kl_fake)
                reg += self.reg_param * avg_kl
                self.update_beta(avg_kl)
            reg.backward()

            d_acc_fake = torch.mean((d_fake < 0.5).float())
            accuracies = {'real': d_acc_real, 'fake': d_acc_fake}

            

            dloss = (dloss_real + dloss_fake)

            clamp_reg_param = max(self.reg_param, 1e-5)
            reg_raw = reg / clamp_reg_param
            self.loss_D = dloss
            D_x, D_G_z1 = dloss_real,dloss_fake
        return D_x, D_G_z1
        
    def em_fn(self, dfake):
        """EM step in our algorithm"""
        with torch.no_grad():
            Dgz = torch.sigmoid(dfake) #1.5 empirically
            while True:
                # gamma_t = np.mean(self.gamma, axis=0, keepdims=True)
                # gamma_ts = np.concatenate([gamma_t for i in range(self.batch_size)], axis=0)
                gamma_ts = self.gamma
                gamma_sum = np.sum(self.gamma, axis=1, keepdims=True)
                gamma_sum = np.concatenate([gamma_sum for i in range(self.num)], axis=1)
                D_f = Dgz.view(-1).reshape(-1, self.batch_size // self.ngpu).detach().cpu().numpy()
                D_f = np.split(D_f, self.ngpu)
                D_ff = np.transpose(np.concatenate([D_f[i] for i in range(self.ngpu)], axis=1))
                w_new = D_ff * np.exp(f.digamma(gamma_ts) - f.digamma(gamma_sum))
                w_normal = w_new / np.sum(w_new, axis=1, keepdims=True)
                gamma_new = self.w + self.alpha
                w_error_value = np.sum(np.abs(self.w - w_normal)) / w_normal.size
                gamma_error_value = np.sum(np.abs(self.gamma - gamma_new)) / gamma_new.size
                if (gamma_error_value < 0.0001) and (w_error_value < 0.0001):
                    break
                else:
                    self.gamma = gamma_new
                    self.w = w_normal


    def backward_G(self):
        """Calculate the gradient of generator"""
        #x_fake = self.G(self.z)
        
        d_fake = self.D(self.fake, self.y)
        if self.network_name == "vgan-based":
            d_fake = d_fake['out']
        self.em_fn(d_fake)
        # add the parameter w into loss
        w_G = self.w / np.sum(self.w, axis=0, keepdims=True)
        w_G = torch.from_numpy(w_G).float().to(self.device)
        w_G = torch.reshape(w_G.transpose(1, 0), [-1, 1])
        if self.network_name == "vgan-based":
            gloss = self.compute_loss(d_fake, 1, w_G)
        else:
            gloss = - torch.mean((torch.reshape(w_G.transpose(1, 0), [-1, 1])).view(-1) * d_fake.view(-1))
        gloss.backward()
        D_G_z2 = d_fake.view(-1).mean().item()
        return D_G_z2

    def optimize_parametersD(self):
        """Optimize the parameters of discriminator"""
        self.forward()
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        D_x, D_G_z1 = self.backward_D()
        self.optimizer_D.step()
        return D_x, D_G_z1
        
    def optimize_parametersG(self):
        """Optimize the parameters of generator"""
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, True)
        self.optimizer_G.zero_grad()
        self.forward()
        D_G_z2 = self.backward_G()
        self.optimizer_G.step()
        if self.network_name == 'vgan-based':
            self.update_average(self.G_test, self.G)
        
        # update parameter alpha
        gamma_sum = np.sum(self.gamma, axis=1, keepdims=True)
        gamma_sum = np.concatenate([gamma_sum for i in range(self.num)], axis=1)
        alpha_lb = np.mean(np.log(f.gamma(np.sum(self.alpha))) - np.sum(np.log(f.gamma(self.alpha))) + \
                           (np.concatenate([self.alpha for i in range(self.batch_size)], 0) - 1) * (
                                   f.digamma(self.gamma) - f.digamma(gamma_sum)))
        alpha_derv = f.digamma(np.sum(self.alpha)) - f.digamma(self.alpha) + \
                     np.mean((f.digamma(self.gamma) - f.digamma(gamma_sum)), axis=0, keepdims=True)
        # print(alpha)
        self.alpha = self.al_optim.step(alpha_derv, self.alpha, self.lr)
        self.alpha = np.clip(self.alpha, 0.1, np.max(self.alpha))
        return D_G_z2

    def update_beta(self, avg_kl):
        """Update beta, when use vgan"""
        with torch.no_grad():
            new_beta = self.reg_param - self.beta_step * (self.target_kl - avg_kl)
            new_beta = max(new_beta, 0)
            # print('setting beta from %.2f to %.2f' % (self.reg_param, new_beta))
            self.reg_param = new_beta
    
    def compute_loss(self, d_out, target, w = torch.Tensor([1])):
        """Return a loss"""
        w = w.to(d_out.device)
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out,targets,reduction='none') * w.view(-1)
        # loss = (2*target - 1) * d_out * w
        return loss.mean()
        
    def update_average(self, model_tgt, model_src, beta=0.999):
        """Moving average"""
        self.set_requires_grad(model_src, False)
        self.set_requires_grad(model_tgt, False)

        param_dict_src = dict(model_src.named_parameters())

        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src) 
