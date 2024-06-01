# %%

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
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

from cyclegan import Generator, Discriminator, TwoDomainDataset, CycleGAN

def test_generator():
  G = Generator(block=2, hidden=8, stage=2, act='relu', name="G")
  sample = jnp.asarray(np.random.normal(0, 1, (8, 16, 16, 3)))
  params = nj.init(G)({}, sample, seed=0)
  fn = jax.jit(nj.pure(G))
  _, out = fn(params, sample)
  assert sample.shape == out.shape


def test_discriminator():
  D = Discriminator(hidden=8, stage=2, name="D")
  sample = jnp.asarray(np.random.normal(0, 1, (8, 16, 16, 3)))
  params = nj.init(D)({}, sample, seed=0)
  fn = jax.jit(nj.pure(D))
  _, out = fn(params, sample)
  assert out.shape == (8, 4, 4, 1)

def test_two_domain_dataset():
  dataset = TwoDomainDataset("../data/monet_jpg", "../data/photo_jpg", (256, 256)).dataset()
  data = next(dataset)
  assert "A" in data and "B" in data
  assert data["A"].shape == (256, 256, 3)
  assert data["B"].dtype == np.uint8
  assert data["A"].shape == (256, 256, 3)
  assert data["B"].dtype == np.uint8


config = embodied.Config(
  seed=42,
  generator=dict(
    block = 2,
    hidden = 16,
    stage = 5,
    act = 'relu',
    norm = 'instance'
  ),
  discriminator=dict(
    stage = 5,
    hidden = 16,
    act = 'leaky_relu',
    norm = 'instance'
  ),
  opt=dict(
    lr = 3e-4,
  ),
  image_size = [256, 256],
  domain_A = "../data/monet_jpg",
  domain_B = "../data/photo_jpg",
  batch_size = 4
)
np.random.seed(config.seed)
trainer = CycleGAN(config, name="cgan")
dataloader = TwoDomainDataset(config.domain_A, config.domain_B, config.image_size, config.batch_size)



def transform(data):
  return jax.tree.map(lambda x: jnp.asarray(x), data)
dataset = iter(embodied.Prefetch(dataloader.dataset, transform, amount=2))
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

i = 2
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

