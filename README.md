# research

Independent research and technical papers by Fredrik Paulin. Everything here is self-published — drafts, preprints, and empirical studies, in markdown with PDF exports where available. Expect work in progress; status is stated honestly per paper.

## layout

Each project lives in its own folder under `papers/`, with markdown sources at the top level and PDF exports in `export/`. `datasets/` will hold published datasets when there are any.

## papers

### simulated world models

A three-paper set on serving data-poor small businesses with simulation-trained world models. Read the position paper first; the two whitepapers are alternative technical routes to the same destination. All three are proposal drafts (June 2026) — nothing is trained yet, and each paper states the evaluation that would confirm or refute it.

**[Manufacture the data, generalize the model, ground the agent](papers/simulated-world-models-whitepaper/position-paper-simulated-world-models.md)** · position paper · [pdf](papers/simulated-world-models-whitepaper/export/position-paper-simulated-world-models.pdf)

Most of the economy is a long tail of small physical businesses that are individually data-poor but share mechanism — perishability, lead times, labour constraints, the accounting identities that bind them. The paper argues that a calibratable simulation can be made accurate enough to manufacture the missing operational data, that a world model trained on it generalizes across the industry, and that value reaches a specific business through an agentic harness grounding the general model in that business's own sparse, censored data. The load-bearing claim — "accurate enough" — is stated as a falsifiable, measurable property.

**[From simulation to a vertical world model](papers/simulated-world-models-whitepaper/whitepaper-sim-to-worldmodel-pipeline.md)** · technical whitepaper · [pdf](papers/simulated-world-models-whitepaper/export/whitepaper-sim-to-worldmodel-pipeline.pdf)

The engineering companion to the position paper: a four-stage pipeline — generate, aggregate, train, adapt — that turns a mechanistic, parameter-randomized simulation into a training corpus and pretrains a generative world model on it. The model is an autoregressive transformer over a factored event tokenization with a probabilistic numeric head, and the corpus includes uncensored ground truth (true demand, counterfactual lost sales) that no real point-of-sale system records. Includes the corpus schema, training objectives, an evaluation built around held-out parameter regions, and an explicit failure analysis.

**[Predicting in latent space](papers/simulated-world-models-whitepaper/whitepaper-joint-embedding-pipeline.md)** · technical whitepaper · [pdf](papers/simulated-world-models-whitepaper/export/whitepaper-joint-embedding-pipeline.pdf)

The alternative route: a joint-embedding predictive architecture (JEPA) that predicts the representation of the future instead of the next event, discarding the unpredictable surface detail — which customer arrives when — while keeping what a manager needs. The simulation supplies paired censored/uncensored views of every world-state, so de-censoring becomes representation learning rather than explicit estimation. Shares the generator and corpus with the generative pipeline; diverges at training and use, where planning happens in latent space. Honest about the cost: representation collapse is a failure mode the generative route never risks.

### sig-reg dynamics

**[SIGReg in LeWM training: a risk-vs-ceiling story](papers/sig-reg-dynamics/sigreg_dynamics.md)** · empirical study · 2026

An empirical study of the SIGReg / learning-rate interaction in a ViT-Tiny + AR-transformer LeWM on the Two-Room benchmark — twelve cells, roughly 60 training runs. The headline is unflattering for SIGReg: at the 2000-step budget it does not raise the achievable encoder ceiling, but adds a failure probability that climbs with λ and learning rate. Two mechanisms are pinned down — variance concentration and an architectural anisotropy from a 1D positional embedding on a 2D patch grid — and the combined fix (Weak-SIGReg plus a 2D-aware embedding) produces linearly-decodable position the unregularized baseline never had.
