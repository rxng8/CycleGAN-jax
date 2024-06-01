# %%

import sys, pathlib
import numpy as np
import jax
import jax.numpy as jnp
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from ruamel import yaml

from functools import partial as bind

import embodied
from embodied.nn import ninjax as nj
from embodied import nn
from embodied.nn import sg


class Generator(nj.Module):

  block: int = 2
  hidden: int = 16
  stage: int = 2

  def __init__(self, **kw) -> None:
    # NOTE the number of hidden is capped at 256
    self._kw = kw

  def __call__(self, inputs: jax.Array):
    # inputs: (B, H, W, C)
    B, H, W, C = inputs.shape
    # Initial convolution layers
    x = self.get("in", nn.Conv2D, self.hidden, 3, stride=1, pad='same', **self._kw)(inputs)
    # encoding
    _hidden = self.hidden * 2
    for s in range(self.stage):
      x = self.get(f"ds{s}", nn.Conv2D, np.minimum(_hidden, 256), 3, stride=2, pad='same', **self._kw)(x)
      _hidden *= 2
    # residual blocks
    for b in range(self.block):
      x = self.get(f"b{b}", nn.ResidualBlock)(x)
    # upsampling, decoding
    _hidden //= 2
    for s in range(self.stage):
      x = self.get(f"us{s}", nn.Conv2D, np.minimum(_hidden, 256), 3, stride=2, transp=True, pad='same', **self._kw)(x)
    # conv out
    # Initial convolution layers
    kw = {**self._kw, 'act': 'tanh', 'norm': 'none'}
    x = self.get("out", nn.Conv2D, C, 3, stride=1, pad='same', **kw)(x)
    # return
    return x

def test_generator():
  G = Generator(block=2, hidden=8, stage=2, act='relu', name="G")
  sample = jnp.asarray(np.random.normal(0, 1, (8, 16, 16, 3)))
  params = nj.init(G)({}, sample, seed=0)
  fn = jax.jit(nj.pure(G))
  _, out = fn(params, sample)
  assert sample.shape == out.shape

# class DummyTrainer(nj.Module):
#   def __init__(self):
#     self.A = Generator(name="G")
#     self.opt = nn.Optimizer(3e-4, name="O")

#   def train(self):
#     self.opt([self.A], self.loss)

#   def loss(self):
#     L = self.A(jnp.ones((1, 256, 256, 3))) - jnp.ones((1, 256, 256, 3)) * 2
#     return L.mean()

# D = DummyTrainer(name="D")
# params = nj.init(D.train)({}, seed=0)


class Discriminator(nj.Module):

  stage: int = 2
  hidden: int = 16
  act: str = 'leaky_relu'
  norm: str = 'instance'

  def __init__(self, **kw) -> None:
    # NOTE: the number of hidden is capped at 512
    self._kw = kw
    unused_keys = ["act", "pad", "stride", "transp", "hidden", "norm"]
    for uk in unused_keys:
      self._kw.pop(uk, None)

  def output_shape(self, shape: tuple):
    H, W = shape
    return (H // 2**self.stage, W // 2**self.stage)

  def __call__(self, inputs: jax.Array):
    # inputs: (B, H, W, C)
    B, H, W, C = inputs.shape

    # input:
    x = self.get("in", nn.Conv2D, self.hidden, 3, pad='same', stride=2,
      norm='none', act=self.act, **self._kw)(inputs)
    # downblock
    _hidden = self.hidden * 2
    for s in range(self.stage - 1):
      x = self.get(f"s{s}", nn.Conv2D, np.minimum(_hidden, 512), 3, pad='same', stride=2,
        norm=self.norm, act=self.act, **self._kw)(x)
      _hidden *= 2
    # out
    x = self.get(f"out", nn.Conv2D, 1, 3, pad='same', stride=1, act='none')(x)
    return x

def test_discriminator():
  D = Discriminator(hidden=8, stage=2, name="D")
  sample = jnp.asarray(np.random.normal(0, 1, (8, 16, 16, 3)))
  params = nj.init(D)({}, sample, seed=0)
  fn = jax.jit(nj.pure(D))
  _, out = fn(params, sample)
  assert out.shape == (8, 4, 4, 1)

# class DummyTrainer(nj.Module):
#   def __init__(self):
#     self.A = Discriminator(name="G")
#     self.opt = nn.Optimizer(3e-4, name="O")

#   def train(self):
#     self.opt([self.A], self.loss)

#   def loss(self):
#     L = self.A(jnp.ones((1, 256, 256, 3))) - jnp.ones((1, 64, 64, 1)) * 2
#     return L.mean()

# D = DummyTrainer(name="D")
# params = nj.init(D.train)({}, seed=0)



class TwoDomainDataset:
  def __init__(self, path_A: str|pathlib.Path|embodied.Path, path_B: str|pathlib.Path|embodied.Path, image_size=(256, 256), batch_size=16) -> None:
    self.path_A = pathlib.Path(path_A)
    self.path_B = pathlib.Path(path_B)
    self.H = image_size[1]
    self.W = image_size[0]
    self._batch_size = batch_size

    # get all image path
    self.domain_A = []
    for fname in os.listdir(self.path_A):
      if self._check_image(fname):
        self.domain_A.append(self.path_A / fname)
    self._len_A = len(self.domain_A)

    self.domain_B = []
    for fname in os.listdir(self.path_B):
      if self._check_image(fname):
        self.domain_B.append(self.path_B / fname)
    self._len_B = len(self.domain_B)

  def _check_image(self, fname: str):
    accepted = [".png", ".jpg", ".jpeg"]
    for a in accepted:
      if fname.endswith(a):
        return True
    return False

  def _sample_one(self):
    iA = np.random.randint(0, self._len_A)
    iB = np.random.randint(0, self._len_B)
    img_A = np.asarray(Image.open(self.domain_A[iA]))
    img_A = np.asarray(cv2.resize(img_A, (self.W, self.H))) if (self.W, self.H) == img_A.shape[:2] else img_A
    img_B = np.asarray(Image.open(self.domain_B[iB]))
    img_B = np.asarray(cv2.resize(img_B, (self.W, self.H))) if (self.W, self.H) == img_B.shape[:2] else img_B
    return {"A": img_A, "B": img_B}

  def _sample(self):
    batch = []
    for _ in range(self._batch_size):
      data = self._sample_one()
      batch.append(data)
    return {k: np.stack([batch[i][k] for i in range(self._batch_size)], 0) for k in batch[0].keys()}

  def dataset(self):
    while True:
      yield self._sample()

def test_two_domain_dataset():
  dataset = TwoDomainDataset("../data/monet_jpg", "../data/photo_jpg", (256, 256)).dataset()
  data = next(dataset)
  assert "A" in data and "B" in data
  assert data["A"].shape == (256, 256, 3)
  assert data["B"].dtype == np.uint8
  assert data["A"].shape == (256, 256, 3)
  assert data["B"].dtype == np.uint8


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


config = embodied.Config(
  seed=42,
  generator=dict(
    block = 3,
    hidden = 32,
    stage = 5,
    act = 'relu',
    norm = 'instance'
  ),
  discriminator=dict(
    stage = 5,
    hidden = 32,
    act = 'leaky_relu',
    norm = 'instance'
  ),
  opt=dict(
    lr = 3e-4,
  ),
  image_size = [256, 256],
  domain_A = "../data/monet_jpg",
  domain_B = "../data/photo_jpg",
  batch_size = 16
)
np.random.seed(config.seed)
trainer = CycleGAN(config, name="cgan")
dataloader = TwoDomainDataset(config.domain_A, config.domain_B, config.image_size, config.batch_size)
def transform(data):
  return jax.tree.map(lambda x: jnp.asarray(x), data)
dataset = iter(embodied.Prefetch(dataloader.dataset, transform))
params = nj.init(trainer.train)({}, next(dataset), seed=np.random.randint(0, 2**16))
train = jax.jit(nj.pure(trainer.train))
losses = []

# %%

for i in range(2000):
  params, (outs, mets) = train(params, next(dataset), seed=np.random.randint(0, 2**16))
  loss = mets['opt_loss']
  losses.append(loss)
  if i % 100 == 0:
    print(f"[Step {i+1}] loss: {loss}")

plt.plot(losses)


# %%

untransform = lambda x: x / 2 + 0.5

i = 0
fig = plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(untransform(outs["real_A"][i]))
plt.subplot(2, 2, 2)
plt.imshow(untransform(outs["fake_B"][i]))
plt.subplot(2, 2, 3)
plt.imshow(untransform(outs["real_B"][i]))
plt.subplot(2, 2, 4)
plt.imshow(untransform(outs["fake_A"][i]))
plt.show()

# %%

plt.imshow(untransform(outs["fake_B"][i]))


# %%

data = next(dataset)

# %%

i = 0
fig = plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(data["A"][i])
plt.subplot(2, 2, 2)
plt.imshow(data["B"][i])
plt.show()

