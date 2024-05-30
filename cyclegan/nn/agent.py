import os
import jax
import jax.numpy as jnp
import numpy as np

from . import nets
from . import jaxutils
from . import ninjax as nj
from .jaxagent import JAXAgentWrapper

tree_map = jax.tree_util.tree_map
tree_flatten = jax.tree_util.tree_flatten


@JAXAgentWrapper
class Agent(nj.Module):
  def __init__(self, obs_space, config) -> None:
    self.obs_space = obs_space
    self.config = config
    self.opt = jaxutils.Optimizer(**config.opt, name="opt")
    self.sample_net = nets.Linear(10, name="sample_net")

  def infer_initial(self, batch_size):
    return {}

  def train_initial(self, batch_size):
    return {}

  def infer(self, data, state, mode="train"):
    return {}, {}

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    modules = [self.sample_net]
    mets, (outs, state, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    outs = {}
    return outs, state, metrics

  def loss(self, data, state):
    x = self.sample_net(data["meow"])
    loss = ((1.0 - x)**2).sum(-1)
    loss = loss.mean()
    outs, state, metrics = {}, {}, {"loss": loss}
    return loss, (outs, state, metrics)

  def report(self, data):
    return {}

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    return obs