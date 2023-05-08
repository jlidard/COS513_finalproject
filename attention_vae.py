
import torch
from torch import nn
from torch.nn import functional as F
from typing import *
from torch import Tensor
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical

from mlp import MLP

from utils import l2_loss, mon_loss

class AttentionVAE(nn.Module):

    def __init__(self,
                 sequence_length: int,
                 num_agents: int,
                 latent_dim: int,
                 embedding_dim: int,
                 num_samples: int = 6,
                 discrete: bool = False,
                 **kwargs) -> None:
        super(AttentionVAE, self).__init__()

        out_dim = sequence_length*2*num_agents
        self.temperature = 1

        self.discrete = discrete
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.num_agents = num_agents
        self.sequence_length = sequence_length

        self.mu_dim = latent_dim
        self.sigma_dim = latent_dim
        self.latent_dim = latent_dim

        self.num_samples = num_samples

        self.mlp_classifier = MLP(input_size=sequence_length*7*num_agents, output_size=self.latent_dim)

        self.input_encoder = MLP(input_size=7, output_size=self.embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=12, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=12, batch_first=True)
        self.fc_mu = MLP(self.embedding_dim, self.mu_dim*num_samples)
        self.fc_sigma = MLP(self.embedding_dim, self.sigma_dim*num_samples)

        self.decoder_layer = MLP(self.latent_dim, self.out_dim)

    def set_temp(self, temp):
        self.temperature = temp


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B, A, T, D]
        :return: (Tensor) List of latent codes
        """

        # input encoding
        batch_size, num_agents, timesteps, traj_dim = input.shape
        input = input.permute(0, 2, 1, 3)  # [B, T, A, C]  #.reshape(batch_size, timesteps, -1)
        input = self.input_encoder(input)

        # transformer temporal encoder (self-attention)
        input = input.permute(0, 2, 1, 3).reshape(batch_size*num_agents, timesteps, -1)  # [B, A, T, C]
        result = self.encoder(input)

        # temporal max pool
        result = torch.max(result, dim=1).values

        # agent cross attention (w/ max pool)
        result = result.reshape(batch_size, num_agents, -1)
        result = self.attention(key=result, query=result, value=result)[0]
        result = torch.max(result, dim=1).values

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)

        if self.discrete:
            mu = mu.reshape(-1, self.num_samples, self.mu_dim)
            mu = self.softmax(mu)
            log_var = None
        else:
            log_var = self.fc_sigma(result).reshape(-1, self.latent_dim)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_layer(z)
        batch_size = result.shape[0]
        result = result.reshape(batch_size, self.num_agents, self.sequence_length, 2, -1)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def discrete_reparameterize(self, p):
        if self.training:
            # At training time we sample from a relaxed Gumbel-Softmax Distribution. The samples are continuous but when we increase the temperature the samples gets closer to a Categorical.
            m = RelaxedOneHotCategorical(self.temperature, p)
            return m.rsample()
        else:
            # At testing time we sample from a Categorical Distribution.
            m = OneHotCategorical(p)
            return m.sample()

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        shortcut = False
        if shortcut:
            batch_size = input.shape[0]
            z = self.mlp_classifier(input.reshape(batch_size, -1)).unsqueeze(1)
            z = self.softmax(z)
            mu = torch.ones_like(z)
            log_var = None
        else:
            mu, log_var = self.encode(input)
            if not self.discrete:
                z = self.reparameterize(mu, log_var)
            else:
                z = self.discrete_reparameterize(mu)
        result = self.decode(z)
        return [result, mu, log_var, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[1]
        input = args[0]
        mu = args[2]
        log_var = args[3]

        kld_weight = 0 #kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss, ind, _ = mon_loss(input[..., 1:3], recons)

        mus = [mu[b, i] for b,i in enumerate(ind)]
        mus = torch.stack(mus, 0)
        min_mu = mus

        if not self.discrete:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        else:
            kld_loss = torch.sum(min_mu*torch.log(min_mu + 1e-6), -1)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach(), "mu": min_mu}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]