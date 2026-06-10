# geometry is the forecast (skerry)

**[Geometry is the forecast: a position for latent world-model planning in archipelago sail navigation](geometry-is-the-forecast.md)** · position paper · v0.3, June 2026

In confined coastal waters — archipelagos, sounds, fjords — the wind a boat actually experiences is dominated by sub-grid interactions with local geometry that no forecast resolves. Weather routing assumes a forecast-given wind field; learned navigation has converged on latent world models that plan in mazes but never in a domain where the environment's dynamics field, not its obstacle map, is the thing to predict. This paper argues the two failures are one opportunity: the archipelago wind field is a systematic, learnable function of geometry, and a joint-embedding predictive world model with latent MPC is the right tool to learn it jointly with vessel dynamics.

It surveys prior work across latent world-model planning, autonomous sailing, weather routing, terrain-conditioned wind downscaling, and maritime trajectory learning; locates the gap as a single empty cell (environment-learned × marine); and proposes a concrete, single-researcher research program — chart-derived environments, a minimal sail simulator, a JEPA world model, and a fixed benchmark — designed to be falsifiable on two consumer GPUs. The load-bearing practical insight: indexed-color raster nautical charts are semantic segmentation by construction, so real coastal geometry comes nearly free.

This is the applied, marine instance of the world-model thread in this repository: it is PLDM's environment class with real coastal geometry and a wind field in place of walls, and it starts from the training recipe in the [sig-reg-dynamics](../sig-reg-dynamics/) study.

Markdown only — no PDF export yet.
