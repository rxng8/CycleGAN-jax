# %%


import numpy as np
import jax
import jax.numpy as jnp

from ruamel import yaml

from embodied.nn import ninjax as nj
from embodied import nn

# class ResidualBlock(nj.Module):
#   def __init__(self, dim: int) -> None:
#     self._dim = dim

#   def __call__(self, inputs: jax.Array, time_embed: jax.Array = None) -> jax.Array:
#     x = self.get("conv", nn.Conv2D, self._dim, 3, stride=1,
#       transp=False, act='silu', norm='layer', pad='same', preact=True)(inputs)
#     if time_embed is not None:
#       t = jax.nn.silu(time_embed)
#       t = self.get("time", nn.Linear, 2 * self._dim)(t)
#       t = t[:, None, None, :]
#       shift, scale = jnp.split(t, 2, axis=-1)
#       x = x * (1 + scale) + shift
#     x = self.get("conv2", nn.Conv2D, self._dim, 3, stride=1,
#       transp=False, act='silu', norm='layer', pad='same', preact=True)(x)
#     res = self.get("res", nn.Conv2D, self._dim, 1, stride=1,
#       transp=False, act='none', norm='none', pad='same')(inputs)
#     return x + res

class ResidualBlock(nj.Module):

  act: str = "relu"
  norm: str = "instance"

  def __init__(self, dim: int) -> None:
    self._dim = dim

  def __call__(self, x: jax.Array) -> jax.Array:
    jnp.pad(x, ,mode='reflect')
    res = self.get("res1", nn.Conv2D, self._dim, 3, stride=1,
      transp=False, act=self.act, norm=self.norm, pad='same')(x)
    res = self.get("res2", nn.Conv2D, self._dim, 3, stride=1,
      transp=False, act=self.act, norm=self.norm, pad='same')(x)

    res = self.get("res", nn.Conv2D, self._dim, 1, stride=1,
      transp=False, act='none', norm='none', pad='same')(inputs)
    return x + res




