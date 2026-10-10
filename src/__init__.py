"""texture_pp: texture super-resolution pipeline (inference only).

Given a low-resolution OBJ mesh texture, renders the model from multiple
per-island / seam views, super-resolves each view with a mask-conditioned
stable-diffusion model, and gathers the results back into texture space to
produce a 4x super-resolved texture.
"""

__version__ = "0.1.0"
