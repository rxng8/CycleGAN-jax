
from typing import List, Dict, Tuple, Callable, Any, Union, Iterable
import jax
import jax.numpy as jnp
import numpy as np

ParamsType = Dict[str, Iterable['Dict[str, jax.Array]']]
"""All neural network train state, optimizer state, parameters, weights are included here
"""

RSSMState = Dict[str, jax.Array]
"""RSSMState is composed of: {

}
"""

ObservationType = Dict[str, jax.Array] | Dict[str, np.ndarray]
"""dictionary of observation
"""

AgentStateType = Tuple[Dict[str, jax.Array], RSSMState] | Dict[str, jax.Array]
"""This is the agent global state through Markov time step for operating in the environment

"""

AuxiliaryDataType = Any
"""Can be anything
"""

MetricsType = Dict[str, jax.Array]
"""Metrics for logging, often included in loss functions, optimizer call, and other modules.
"""

OptimizerOutputType = Tuple[ParamsType, MetricsType, AuxiliaryDataType] | Tuple[ParamsType, MetricsType]

LossOutputType = Tuple[jax.Array, Tuple[AgentStateType, AuxiliaryDataType, MetricsType]]
""" The output of every loss function that include a constant and a tuple
  of auxiliary data: (agent state, out, metrics)
"""

TrainingOutputType = Tuple[AgentStateType, AuxiliaryDataType, MetricsType]

