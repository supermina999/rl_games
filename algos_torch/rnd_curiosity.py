import torch
from torch import nn
import numpy as np

class RNDCuriosityNetwork(nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def forward(self, obs):
        rnd_res, res = self.network(obs)
        loss = ((res - rnd_res)**2).mean(dim=1)
        return loss



class RNDCurisityTrain:
    def __init__(self, state_shape, model, config, writter, _preproc_obs):
        rnd_config = {
            'input_shape' : state_shape,
        } 
        self.model = RNDCuriosityNetwork(model.build('rnd', **rnd_config)).cuda()
        self.config = config
        self.lr = config['lr']
        self.writter = writter
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.lr))
        self._preproc_obs = _preproc_obs
        self.frame = 0

    def get_loss(self, obs):
        self.model.eval()
        with torch.no_grad():
            return self.model(obs).detach().cpu().numpy() * self.config['scale_value']

    def train(self, obs):
        self.model.train()
        mini_epoch = self.config['mini_epochs']
        mini_batch = self.config['minibatch_size']

        num_minibatches = np.shape(obs)[0] // mini_batch
        self.frame = self.frame + 1
        for _ in range(mini_epoch):
            # returning loss from last epoch
            avg_loss = 0
            for i in range(num_minibatches):
                obs_batch = obs[i * mini_batch: (i + 1) * mini_batch]
                obs_batch = self._preproc_obs(obs_batch)
                loss = self.model(obs_batch).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

        self.writter.add_scalar('rnd/train_loss', avg_loss, self.frame)
        return avg_loss / num_minibatches