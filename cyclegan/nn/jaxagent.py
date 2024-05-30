import os
import jax
import jax.numpy as jnp
import numpy as np

from ..core import Agent, Counter, Batcher, when
from . import jaxutils
from . import ninjax as nj
from . import nets

tree_map = jax.tree_util.tree_map
tree_flatten = jax.tree_util.tree_flatten


def JAXAgentWrapper(agent_cls):
  class Agent(JAXAgent):
    inner = agent_cls
    def __init__(self, *args, **kwargs):
      super().__init__(agent_cls, *args, **kwargs)
  return Agent


class JAXAgent(Agent):

  def __init__(self, agent_cls, obs_space, config):
    self.config = config
    self.jaxconfig = config.jax
    self._setup()
    self.agent = agent_cls(obs_space, config, name='agent')
    self.rng = np.random.default_rng(config.seed)

    available = jax.devices(self.jaxconfig.platform)
    self.infer_devices = [available[i] for i in self.jaxconfig.infer_devices]
    self.train_devices = [available[i] for i in self.jaxconfig.train_devices]
    self.single_device = (self.infer_devices == self.train_devices) and (
        len(self.infer_devices) == 1)
    print(f'JAX devices ({jax.local_device_count()}):', available)
    print('Policy devices:', ', '.join([str(x) for x in self.infer_devices]))
    print('Train devices: ', ', '.join([str(x) for x in self.train_devices]))

    self._once = True
    self._updates = Counter()
    self._should_metrics = when.Every(self.jaxconfig.metrics_every)
    self._transform()
    self.varibs = self._init_varibs(obs_space)
    self.sync()

  def infer(self, obs, state=None, mode='train'):
    obs = obs.copy()
    obs = self._convert_inps(obs, self.infer_devices)
    varibs = self.varibs if self.single_device else self.infer_varibs
    if state is None:
      bs_ones = self._convert_inps(np.ones((self.config.batch_size,)), self.infer_devices)
      _, state = self._init_infer(varibs, bs_ones)
    else:
      state = tree_map(
          np.asarray, state, is_leaf=lambda x: isinstance(x, list))
      state = self._convert_inps(state, self.infer_devices)
    _, (outs, state) = self._infer(varibs, obs, state, mode=mode)
    outs = self._convert_outs(outs, self.infer_devices)
    # TODO: Consider keeping policy states in accelerator memory.
    state = self._convert_outs(state, self.infer_devices)
    return outs, state

  def train(self, data, state=None):
    if state is None:
      bs_ones = self._convert_inps(np.ones((self.config.batch_size,)), self.train_devices)
      self.varibs, state = self._init_train(self.varibs, bs_ones)
    self.varibs, (outs, state, mets) = self._train(
        self.varibs, data, state)
    outs = self._convert_outs(outs, self.train_devices)
    self._updates.increment()
    if self._should_metrics(self._updates):
      mets = self._convert_mets(mets, self.train_devices)
    else:
      mets = {}
    # assert if we have successfully trained one step
    # This required the Optimizer in jaxutils counted.
    if self._once:
      self._once = False
      assert jaxutils.Optimizer.PARAM_COUNTS
      for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
        mets[f'params_{name}'] = float(count)
    return outs, state, mets

  def report(self, data):
    rng = self._next_rngs(self.train_devices)
    mets, _ = self._report(self.varibs, rng, data)
    mets = self._convert_mets(mets, self.train_devices)
    return mets

  def dataset(self, generator):
    batcher = Batcher(
        sources=[generator] * self.config.batch_size,
        workers=self.config.workers,
        postprocess=lambda x: self._convert_inps(x, self.train_devices),
        prefetch_source=4, prefetch_batch=1)
    return batcher()

  def save(self):
    if len(self.train_devices) > 1:
      varibs = tree_map(lambda x: x[0], self.varibs)
    else:
      varibs = self.varibs
    varibs = jax.device_get(varibs)
    data = tree_map(np.asarray, varibs)
    return data

  def load(self, state):
    if len(self.train_devices) == 1:
      self.varibs = jax.device_put(state, self.train_devices[0])
    else:
      self.varibs = jax.device_put_replicated(state, self.train_devices)
    self.sync()

  def sync(self):
    if self.single_device:
      return
    if len(self.train_devices) == 1:
      varibs = self.varibs
    else:
      varibs = tree_map(lambda x: x[0].device_buffer, self.varibs)
    if len(self.infer_devices) == 1:
      self.infer_varibs = jax.device_put(varibs, self.infer_devices[0])
    else:
      self.infer_varibs = jax.device_put_replicated(
          varibs, self.infer_devices)

  def _setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
    except Exception as e:
      print('Could not disable TensorFlow devices:', e)
    if not self.jaxconfig.prealloc:
      os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    xla_flags = []
    if self.jaxconfig.logical_cpus:
      count = self.jaxconfig.logical_cpus
      xla_flags.append(f'--xla_force_host_platform_device_count={count}')
    if xla_flags:
      os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
    jax.config.update('jax_platform_name', self.jaxconfig.platform)
    jax.config.update('jax_disable_jit', not self.jaxconfig.jit)
    jax.config.update('jax_debug_nans', self.jaxconfig.debug_nans)
    jax.config.update('jax_transfer_guard', 'disallow')
    if self.jaxconfig.platform == 'cpu':
      jax.config.update('jax_disable_most_optimizations', self.jaxconfig.debug)
    jaxutils.COMPUTE_DTYPE = getattr(jnp, self.jaxconfig.precision)

  def _transform(self):
    self._init_infer = nj.pure(lambda x: self.agent.infer_initial(len(x))) # NOTE: jax.jit pure function convention
    self._init_train = nj.pure(lambda x: self.agent.train_initial(len(x))) # NOTE: jax.jit pure function convention
    self._infer = nj.pure(self.agent.infer)
    self._train = nj.pure(self.agent.train)
    self._report = nj.pure(self.agent.report)
    if len(self.train_devices) == 1:
      kw = dict(device=self.train_devices[0])
      self._init_train = jax.jit(self._init_train, **kw)
      self._train = jax.jit(self._train, **kw)
      self._report = jax.jit(self._report, **kw)
    else:
      kw = dict(devices=self.train_devices)
      self._init_train = jax.pmap(self._init_train, 'i', **kw)
      self._train = jax.pmap(self._train, 'i', **kw)
      self._report = jax.pmap(self._report, 'i', **kw)
    if len(self.infer_devices) == 1:
      kw = dict(device=self.infer_devices[0])
      self._init_infer = jax.jit(self._init_infer, **kw)
      self._infer = jax.jit(self._infer, static_argnames=['mode'], **kw)
    else:
      kw = dict(devices=self.infer_devices)
      self._init_infer = jax.pmap(self._init_infer, 'i', **kw)
      self._infer = jax.pmap(self._infer, 'i', static_argnames=['mode'], **kw)

  def _convert_inps(self, value, devices):
    if len(devices) == 1:
      value = jax.device_put(value, devices[0])
    else:
      check = tree_map(lambda x: len(x) % len(devices) == 0, value)
      if not all(jax.tree_util.tree_leaves(check)):
        shapes = tree_map(lambda x: x.shape, value)
        raise ValueError(
            f'Batch must by divisible by {len(devices)} devices: {shapes}')
      # TODO: Avoid the reshape?
      value = tree_map(
          lambda x: x.reshape((len(devices), -1) + x.shape[1:]), value)
      shards = []
      for i in range(len(devices)):
        shards.append(tree_map(lambda x: x[i], value))
      value = jax.device_put_sharded(shards, devices)
    return value

  def _convert_outs(self, value, devices):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if len(devices) > 1:
      value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
    return value

  def _convert_mets(self, value, devices):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if len(devices) > 1:
      value = tree_map(lambda x: x[0], value)
    return value

  def _next_rngs(self, devices, mirror=False, high=2 ** 63 - 1):
    if len(devices) == 1:
      return jax.device_put(self.rng.integers(high), devices[0])
    elif mirror:
      return jax.device_put_replicated(
          self.rng.integers(high), devices)
    else:
      return jax.device_put_sharded(
          list(self.rng.integers(high, size=len(devices))), devices)

  def _init_varibs(self, obs_space):
    varibs = {}
    rng = self._next_rngs(self.train_devices, mirror=True)
    dims = (self.config.batch_size,)
    data = self._dummy_batch(obs_space, dims)
    data = self._convert_inps(data, self.train_devices)
    bs_ones = self._convert_inps(np.ones((self.config.batch_size,)), self.train_devices)
    varibs = nj.init(lambda x: self.agent.train_initial(len(x)))(varibs, bs_ones, seed=rng)
    _, state = self._init_train(varibs, bs_ones)
    varibs = nj.init(self.agent.train)(varibs, data, state, seed=rng)
    return varibs

  def _dummy_batch(self, spaces, batch_dims):
    spaces = list(spaces.items())
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data


@JAXAgentWrapper
class SampleAgent(nj.Module):
  def __init__(self, obs_space, config) -> None:
    self.obs_space = obs_space
    self.config = config
    self.opt = jaxutils.Optimizer(**config.opt, name="opt")
    self.sample_net = nets.Linear(10, name="sample_net")

  def infer_initial(self, batch_size):
    return {}

  def train_initial(self, batch_size):
    self.get("woofwoof", jnp.ones, (3,))
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
    return obs