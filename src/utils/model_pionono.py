import torch
from torch import nn
import utils.globals
import numpy as np
from Probabilistic_Unet_Pytorch.utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
from torch.distributions import Normal, Independent, kl
from utils.model_headless import UnetHeadless
from segmentation_models_pytorch import Unet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LatentVariable(nn.Module):
    """

    """
    def __init__(self, num_annotators, latent_dims=2, prior_mu_value=0.0, prior_sigma_value=1.0):
        super(LatentVariable, self).__init__()
        self.latent_dims = latent_dims
        self.no_annotators = num_annotators
        prior_mu, prior_cov = self._init_distributions(prior_mu=prior_mu_value, prior_sigma=prior_sigma_value)
        self.prior_mu = torch.nn.Parameter(prior_mu)
        self.prior_cov = torch.nn.Parameter(prior_cov)
        self.prior_mu.requires_grad = False
        self.prior_cov.requires_grad = False
        posterior_mu, posterior_cov = self._init_distributions(prior_mu=prior_mu_value, prior_sigma=prior_sigma_value)
        self.posterior_mu = torch.nn.Parameter(posterior_mu)
        self.posterior_cov = torch.nn.Parameter(posterior_cov)
        self.name = 'LatentVariable'

    def _init_distributions(self, prior_mu=0.0, prior_sigma=1.0, trainable=True):
        mu_list = []
        cov_list = []
        for a in range(self.no_annotators):
            if isinstance(prior_mu, list):
                mu = prior_mu[a]
                sigma = prior_sigma[a]
            else:
                mu = prior_mu
                sigma = prior_sigma
            mu_a = np.ones(self.latent_dims)*mu
            cov_a = np.eye(self.latent_dims) * (sigma*sigma)
            mu_list.append(mu_a)
            cov_list.append(cov_a)
        mu_list = torch.tensor(np.array(mu_list))
        cov_list = torch.tensor(np.array(cov_list))
        return mu_list, cov_list

    def forward(self, annotator, sample=True):
        z = torch.zeros([len(annotator), self.latent_dims]).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            dist_a = torch.distributions.multivariate_normal.MultivariateNormal(self.posterior_mu[a],
                                                                                scale_tril=torch.tril(self.posterior_cov[a]))

            if sample:
                z_i = dist_a.rsample()
            else:
                z_i = dist_a.loc
            z[i] = z_i
        return z

    def get_kl_loss(self, annotator):
        kl_loss = torch.zeros([len(annotator)]).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            dist_a_posterior = torch.distributions.multivariate_normal.MultivariateNormal(self.posterior_mu[a],
                                                                                          scale_tril=torch.tril(self.posterior_cov[a]))
            dist_a_prior = torch.distributions.multivariate_normal.MultivariateNormal(self.prior_mu[a],
                                                                                scale_tril=torch.tril(self.prior_cov[a]))

            kl_loss[i] = torch.distributions.kl_divergence(dist_a_posterior, dist_a_prior)
        kl_mean = torch.mean(kl_loss)
        return kl_mean


class MyFcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters_last_layer, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(MyFcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters_last_layer = num_filters_last_layer
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters_last_layer+self.latent_dim, self.num_filters_last_layer, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters_last_layer, self.num_filters_last_layer, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv2d(self.num_filters_last_layer, self.num_classes, kernel_size=1)
            self.activation = torch.nn.Softmax(dim=1)

            self.layers.apply(self.initialize_weights)
            self.last_layer.apply(self.initialize_weights)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            x = self.layers(feature_map)
            x = self.last_layer(x)
            y = self.activation(x)
            return y

class PiononoModel(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=3, num_classes=1, num_annotators=6, predict_annotator=0, latent_dim=6,
                 z_prior_mu=0.0, z_prior_sigma=1.0, no_convs_fcomb=4, kl_factor=1.0, reg_factor=0.1):
        super(PiononoModel, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.predict_annotator = predict_annotator
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.kl_factor = kl_factor
        self.reg_factor = reg_factor
        self.unet = UnetHeadless().to(device)
        self.z = LatentVariable(num_annotators, latent_dim, prior_mu_value=z_prior_mu, prior_sigma_value=z_prior_sigma).to(device)
        self.fcomb = MyFcomb(16, self.latent_dim, self.input_channels, self.num_classes,
                             self.no_convs_fcomb, {'w' :'orthogonal', 'b' :'normal'}, use_tile=True).to(device)

    # TODO: move annotator to sample function
    def forward(self, patch):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        self.unet_features = self.unet.forward(patch)

    def sample(self, use_z_mean: bool = False, annotator: torch.tensor = None):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if annotator is None:
            annotator = torch.ones(self.unet_features.shape[0]).to(device) * self.predict_annotator
        self.current_annotator = annotator

        if use_z_mean == False:
            z = self.z.forward(self.current_annotator, sample=True)
        else:
            z = self.z.forward(self.current_annotator, sample=False)
        pred = self.fcomb.forward(self.unet_features, z)

        return pred

    def elbo(self, labels: torch.tensor, loss_fct, annotator: torch.tensor):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        self.preds = self.sample(use_z_mean=False, annotator=annotator)
        self.log_likelihood_loss = loss_fct(self.preds, labels)
        self.kl_loss = self.z.get_kl_loss(annotator) * self.kl_factor

        return -(self.log_likelihood_loss + self.kl_loss)

    def combined_loss(self, labels, loss_fct, annotator):
        elbo = self.elbo(labels, loss_fct=loss_fct, annotator=annotator)
        self.reg_loss = l2_regularisation(self.fcomb.layers) * self.reg_factor
        loss = -elbo + self.reg_loss
        return loss