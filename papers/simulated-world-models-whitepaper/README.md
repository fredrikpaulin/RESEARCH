# simulated world models

Three papers on manufacturing training data from mechanistic simulation and spending the resulting world model on one real small business at a time. Proposal drafts, June 2026 — nothing is trained yet; each paper specifies the evaluation that would confirm or refute it.

## reading order

1. **[Manufacture the data, generalize the model, ground the agent](position-paper-simulated-world-models.md)** ([pdf](export/position-paper-simulated-world-models.pdf)) — the position paper. Why a data-poor but structure-rich industry can have its missing data manufactured from simulation, and why value reaches a specific business through an agentic harness rather than the model alone. Read this first.

2. **[From simulation to a vertical world model](whitepaper-sim-to-worldmodel-pipeline.md)** ([pdf](export/whitepaper-sim-to-worldmodel-pipeline.pdf)) — the default technical route. Generative: tokenize events, predict the next one. Corpus schema, architecture, training objectives, evaluation, failure analysis.

3. **[Predicting in latent space](whitepaper-joint-embedding-pipeline.md)** ([pdf](export/whitepaper-joint-embedding-pipeline.pdf)) — the alternative route. Joint-embedding (JEPA): predict representations, not events; plan in latent space. Shares stages 1–2 with the generative pipeline and includes a comparison of when to prefer which.

The worked vertical throughout is a perishable-goods retailer (a bakery), used as a concrete archetype — nothing in the pipelines is specific to it.
