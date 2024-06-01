"""
File: trainer.py
Author: Viet Nguyen
Date: 2024-06-01

Description: Trainer of CycleGAN model
"""


from ruamel import yaml
import numpy as np
import jax
import jax.numpy as jnp

from functools import partial as bind

import embodied
from embodied.nn import ninjax as nj
from embodied import nn
from embodied.nn import sg

from .nets import Generator, Discriminator


class CycleGAN(nj.Module):

  def __init__(self, config: embodied.Config) -> None:
    self.G_AB = Generator(**config.generator, name="G_AB")
    self.G_BA = Generator(**config.generator, name="G_BA")
    self.D_A = Discriminator(**config.discriminator, name="D_A")
    self.D_B = Discriminator(**config.discriminator, name="D_B")
    self.opt = nn.Optimizer(**config.opt, name="opt")
    self.modules = [self.G_AB, self.G_BA, self.D_A, self.D_B]
    self._image_shape = config.image_size[::-1] # [W, H] -> [H, W]
    self._dis_out = self.D_A.output_shape(self._image_shape)

  def train(self, data: dict):
    mets = {}
    opt_mets, (out, loss_mets) = self.opt(self.modules, self.loss, self.preprocess(data), has_aux=True)
    mets.update(opt_mets)
    mets.update(loss_mets)
    return out, mets

  def preprocess(self, data):
    result = {}
    for key, value in data.items():
      if len(value.shape) >= 3 and value.dtype == jnp.uint8:
        value = (nn.cast(value) / 255.0 - 0.5) * 2
      else:
        raise NotImplementedError("should all be images")
      result[key] = value
    return result

  def loss(self, data: dict) -> tuple:
    real_A = data['A']
    real_B = data['B']
    B, H, W, C = real_A.shape
    valid = jnp.ones((B, *self._dis_out, 1))
    fake = jnp.zeros((B, *self._dis_out, 1))

    losses = {}
    # identity loss
    losses["id_A"] = jnp.abs(self.G_BA(real_A) - real_A) # If put A into G:B->A. Of course it still return A
    losses["id_B"] = jnp.abs(self.G_AB(real_B) - real_B) # If put B into G:A->B. Of course it still return B
    # GAN loss
    fake_B = self.G_AB(real_A)
    fake_A = self.G_BA(real_B)
    losses["gan_AB"] = (self.D_B(fake_B) - valid)**2 # trick discriminator B
    losses["gan_BA"] = (self.D_A(fake_A) - valid)**2 # trick discriminator A
    # Cycle loss
    recov_A = self.G_BA(fake_B)
    recov_B = self.G_AB(fake_A)
    losses["cycle_A"] = jnp.abs(real_A - recov_A)
    losses["cycle_B"] = jnp.abs(real_B - recov_B)
    # discriminator loss
    losses["disc_real_A"] = (self.D_A(real_A) - valid)**2
    losses["disc_fake_A"] = (self.D_A(sg(fake_A)) - fake)**2
    losses["disc_real_B"] = (self.D_B(real_B) - valid)**2
    losses["disc_fake_B"] = (self.D_B(sg(fake_B)) - fake)**2

    model_loss = sum([x.mean() for x in losses.values()])
    mets = self._metrics(data, losses)
    outs = {
      "real_A": real_A,
      "fake_A": fake_A,
      "real_B": real_B,
      "fake_B": fake_B,
    }
    return model_loss, (outs, mets)

  def _metrics(self, data: dict, losses: dict):
    mets = {}
    for k, v in losses.items():
      mets.update(nn.tensorstats(v, prefix=k))
    return mets
