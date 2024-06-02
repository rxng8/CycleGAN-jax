# %%

import numpy as np
import jax
import jax.numpy as jnp

batch = np.random.normal(0, 1, (8, 256, 256, 3))
# batch: (B, H, W, C)
B, H, W, C = batch.shape

_batch = batch[:4]

# %%

int(np.sqrt(B))






# %%


