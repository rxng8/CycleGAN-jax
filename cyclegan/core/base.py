from .config import Config
from .space import Space

class Agent:
  def __init__(self, obs_space: Space, config: Config):
    """

    Args:
        obs_space (_type_): The actual data recieved as input
        config (_type_): configuration
    """
    pass

  def dataset(self, generator_fn):
    raise NotImplementedError(
        'dataset(generator_fn) -> generator_fn')

  def infer(self, obs, state=None, mode='train'):
    raise NotImplementedError(
        "infer(obs, state=None, mode='train') -> out, state")

  def train(self, data, state=None):
    raise NotImplementedError(
        'train(data, state=None) -> outs, state, metrics')

  def report(self, data):
    raise NotImplementedError(
        'report(data) -> metrics')

  def save(self):
    raise NotImplementedError('save() -> data')

  def load(self, data):
    raise NotImplementedError('load(data) -> None')

  def sync(self):
    # This method allows the agent to sync parameters from its training devices
    # to its policy devices in the case of a multi-device agent.
    pass

class Dataset:

  def __len__(self):
    raise NotImplementedError('Returns: total number of steps')

  @property
  def stats(self):
    raise NotImplementedError('Returns: metrics')

  @property
  def obs_space(self):
    # By convention, keys starting with log_ are not consumed by the agent.
    raise NotImplementedError('Returns: dict of spaces')

  def dataset(self):
    raise NotImplementedError('Yields: one instance (unbatched) of dictionary of data')

  def save(self):
    pass

  def load(self, data):
    pass
