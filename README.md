# research

Independent research and technical papers by Fredrik Paulin. Everything here is self-published — drafts, preprints, and empirical studies, in markdown with PDF exports where available. Expect work in progress; status is stated honestly per paper.

## layout

Each project lives in its own folder under `papers/`, with markdown sources at the top level and PDF exports in `export/`. `datasets/` will hold published datasets when there are any.

## papers

The papers form one program rather than a pile: an agent that needs a world model, the world models that fill it across two domains, and the training methods underneath. They're grouped that way below — by theme, not by date.

### agentic systems

The frame the rest plugs into.

**[A curiosity-driven agentic system for knowledge gap identification and targeted exploration](papers/curiosity-driven-agentic-system-for-knowledge-gap-and-targeted-exploration/A_Curiosity_Driven_Agentic_System_for_Knowledge_Gap_Identification_and_Targeted_Exploration.pdf)** · whitepaper · February 2026

LLMs answer fluently whether or not they actually know, and don't reliably mark the boundary. This whitepaper proposes an agentic system that treats "not knowing" as a first-class state: a structured knowledge model partitions facts into known/unknown/uncertain, the LLM does task parsing and gap detection against it, a planning module converts gaps into targeted actions — retrieval, simulation, experimentation, inspection — and a feedback loop writes validated findings back. Covers subsystem interfaces, data flow, constraints, evaluation protocols, and case studies across domains. This is the umbrella architecture the rest of the research plugs into: the world-model work below specifies one of its components.

### world models

The substance — proposals for world models in two domains.

A three-paper set on serving data-poor small businesses with simulation-trained world models. Read the position paper first; the two whitepapers are alternative technical routes to the same destination. All three are proposal drafts (June 2026) — nothing is trained yet, and each paper states the evaluation that would confirm or refute it.

**[Manufacture the data, generalize the model, ground the agent](papers/simulated-world-models-whitepaper/position-paper-simulated-world-models.md)** · position paper · [pdf](papers/simulated-world-models-whitepaper/export/position-paper-simulated-world-models.pdf)

Most of the economy is a long tail of small physical businesses that are individually data-poor but share mechanism — perishability, lead times, labour constraints, the accounting identities that bind them. The paper argues that a calibratable simulation can be made accurate enough to manufacture the missing operational data, that a world model trained on it generalizes across the industry, and that value reaches a specific business through an agentic harness grounding the general model in that business's own sparse, censored data. The load-bearing claim — "accurate enough" — is stated as a falsifiable, measurable property.

**[From simulation to a vertical world model](papers/simulated-world-models-whitepaper/whitepaper-sim-to-worldmodel-pipeline.md)** · technical whitepaper · [pdf](papers/simulated-world-models-whitepaper/export/whitepaper-sim-to-worldmodel-pipeline.pdf)

The engineering companion to the position paper: a four-stage pipeline — generate, aggregate, train, adapt — that turns a mechanistic, parameter-randomized simulation into a training corpus and pretrains a generative world model on it. The model is an autoregressive transformer over a factored event tokenization with a probabilistic numeric head, and the corpus includes uncensored ground truth (true demand, counterfactual lost sales) that no real point-of-sale system records. Includes the corpus schema, training objectives, an evaluation built around held-out parameter regions, and an explicit failure analysis.

**[Predicting in latent space](papers/simulated-world-models-whitepaper/whitepaper-joint-embedding-pipeline.md)** · technical whitepaper · [pdf](papers/simulated-world-models-whitepaper/export/whitepaper-joint-embedding-pipeline.pdf)

The alternative route: a joint-embedding predictive architecture (JEPA) that predicts the representation of the future instead of the next event, discarding the unpredictable surface detail — which customer arrives when — while keeping what a manager needs. The simulation supplies paired censored/uncensored views of every world-state, so de-censoring becomes representation learning rather than explicit estimation. Shares the generator and corpus with the generative pipeline; diverges at training and use, where planning happens in latent space. Honest about the cost: representation collapse is a failure mode the generative route never risks.

And the second domain — the same world-model idea applied at sea:

**[Geometry is the forecast: a position for latent world-model planning in archipelago sail navigation](papers/geometry-is-the-forecast/geometry-is-the-forecast.md)** · position paper · June 2026

Point-to-point sailing in confined coastal waters falls between two research traditions and is served well by neither: weather routing optimizes over forecast grids too coarse to resolve the wind shadow behind a 400 m island, and learned latent world-models plan well in mazes but have never touched a domain where the environment itself — not the obstacle map — must be predicted. The paper argues these are the same opportunity: the wind field in an archipelago is a systematic, learnable function of geometry, and a JEPA world model with latent MPC is the right tool to learn it jointly with vessel dynamics, because ranking routes needs predicted consequences, not a reconstructed weather map. It surveys the intersecting prior work, names the empty cell at the centre, and proposes a program a single researcher with two consumer GPUs could falsify — built on the observation that indexed-palette nautical charts are semantic segmentation by construction, making real coastal environments nearly free. Shares the joint-embedding approach of the simulated-world-models pipelines and builds on the SIGReg training recipe below.

### world-model training

Methods and empirical results — the recipe the world-model proposals above build on.

**[SIGReg in LeWM training: a risk-vs-ceiling story](papers/sig-reg-dynamics/sigreg_dynamics.md)** · empirical study · 2026

An empirical study of the SIGReg / learning-rate interaction in a ViT-Tiny + AR-transformer LeWM on the Two-Room benchmark — twelve cells, roughly 60 training runs. The headline is unflattering for SIGReg: at the 2000-step budget it does not raise the achievable encoder ceiling, but adds a failure probability that climbs with λ and learning rate. Two mechanisms are pinned down — variance concentration and an architectural anisotropy from a 1D positional embedding on a 2D patch grid — and the combined fix (Weak-SIGReg plus a 2D-aware embedding) produces linearly-decodable position the unregularized baseline never had.
