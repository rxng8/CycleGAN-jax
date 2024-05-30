
from typing import List, Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from ..nn import nets, jaxutils, ninjax as nj
from ..nn.tom import TOM
from ..nn.gmm import GMM
from ..nn.jaxagent import JAXAgentWrapper

@JAXAgentWrapper
class Agent(nj.Module):
  def __init__(self, obs_space, config) -> None:
    self.obs_space = obs_space
    self.config = config
    self.gmm = GMM(config, name="gmm")
    self.tom = TOM(config, name="tom")

  def infer_initial(self, batch_size):
    return {}

  def train_initial(self, batch_size):
    return {}

  def preprocess_data(self, data: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
    # convert all the image data into float32 and normalize image
    data = data.copy()
    for key, value in data.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      data[key] = value

    # human segmentation
    parse_array = data["human_segmentation"]

    # cloth parse
    parse_cloth_mask = jaxutils.cast_to_compute(parse_array == 5) + \
      jaxutils.cast_to_compute(parse_array == 6) + \
      jaxutils.cast_to_compute(parse_array == 7)
    parse_cloth_mask = parse_cloth_mask[..., None]
    parse_cloth = parse_cloth_mask * data["human_image"] + (1 - parse_cloth_mask)

    # build person agnostic representation
    person_mask = (parse_array > 1).astype(jnp.float32)[..., None] # (H, W) => (H, W, 1)
    parse_head = jnp.sum(jnp.asarray([parse_array == seg for seg in (1, 2, 4, 13)], dtype=jnp.float32), axis=0) # (seg, B, H, W) => (B, H, W)
    human_head = parse_head[..., None] * data["human_image"] # (B, H, W, C)
    agnostic = jnp.concatenate([person_mask, human_head, data["human_pose_map"]], axis=-1) # (B, H, W, 1 + 3 + 18)
    return {**data,
      "cloth_parse": parse_cloth, # image of the human cloth
      "cloth_parse_mask": parse_cloth_mask, # the mask of the human cloth in the actual human image
      "human_mask": person_mask, # mask of the human in the human image
      "human_head": human_head, # image of human head
      "human_agnostic": agnostic # human agnostic representation
    }

  def infer(self, data, state, mode="infer"):
    outs = self.gmm.infer(self.preprocess_data(data))
    outs = self.tom.infer(outs)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    outs_data, _, opt_gmm_metrics = self.gmm.train(self.preprocess_data(data))
    metrics.update(opt_gmm_metrics)
    outs_data, _, opt_tom_metrics = self.tom.train(jaxutils.sg(outs_data))
    metrics.update(opt_tom_metrics)
    return outs_data, state, metrics

  def report(self, data):
    return {}
