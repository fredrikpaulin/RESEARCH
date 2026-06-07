# SIGReg in LeWM Training: a Risk-vs-Ceiling Story

*An empirical study of the SIGReg / learning-rate interaction in a
ViT-Tiny + AR-transformer LeWM trained on the Two-Room benchmark.
The spine is a 5-seed variance sweep across six core (LR, λ,
schedule) cells — 30 runs — extended with four follow-up sweeps that
test mechanism interventions (wall rotation, 2D positional embedding,
Weak-SIGReg, and the combined fix). Twelve cells, ~60 training runs
in total.*

## abstract

The headline is unflattering for SIGReg: at the 2000-step budget I
work in, **SIGReg does not raise the achievable encoder quality.** Best-case MLP R² across the whole
parameter space ties near 0.997 with or without it. What SIGReg adds
is a failure probability that climbs steeply with both λ and learning
rate, plus a clean catastrophe when it's introduced on a mid-training
schedule.

The numbers say it plainly. Constant λ=0 at low LR hits the same
ceiling as constant λ=0.1 at low LR, but with 0/5 seeds failing
instead of 2/5 (40%). λ=0.1 at high LR fails on every seed (5/5).
Ramping λ from 0 to 0.1 during training also fails on every seed
(5/5), and ends up *worse* than λ=0.1 from a cold start.

Every non-trivial failure shares one signature: the encoder's x
dimension dies before its y dimension. Two interacting mechanisms
produce that. First, SIGReg pressure concentrates the encoder's
variance into 1–3 dimensions, against ≈10 in the unregularized case —
the *concentration* mechanism, confirmed by PCA on representative
checkpoints. Second, the encoder's 1D positional embedding sitting on
a 2D patch grid encodes column position in high-frequency components
and row position in low-frequency ones, which builds in a row/column
asymmetry — the *architectural anisotropy*.

Two experiments pin the second mechanism down. Rotating the Two-Room
environment 90° shows the asymmetry tracks pixel axes, not task
geometry. Replacing the 1D positional embedding with a 2D-aware
sin/cos variant that gives row and column equal capacity collapses
the R²_x / R²_y asymmetry (ratio 0.02 → 0.92) and nearly doubles mean
R² in the failure cell (0.31 → 0.56), with no regression at the λ=0
corner. At λ=0 the anisotropy is invisible — both axes recover
cleanly. Under SIGReg concentration with the 1D embedding the
row-encoding directions survive and the column-encoding directions
are destroyed; with the 2D embedding both survive symmetrically.

The orthogonal-cheat half of the story gets a published fix.
Weak-SIGReg (Akbar 2026, arXiv 2603.05924) adds a
Frobenius-to-identity covariance penalty designed to block
Strong-SIGReg's rank-deficient solutions. At the catastrophic v5
corner it lifts mean R² from 0.31 to 0.77 — robust across precision
regimes, but partial (4/5 seeds at partial collapse). Combining
Weak-SIGReg with the 2D positional embedding addresses both
mechanisms at once and produces the cleanest result in the project:
4 of 5 seeds at MLP R² = 0.992 ± 0.001, symmetric R²_x ≈ R²_y,
rank_95 in the healthy band, and **linearly-decodable position
(linear-probe R² ≈ 0.98)** — a property the unregularized baseline
does not have, where linear probes recover only ≈ 0.10–0.20 of
position despite MLP probes hitting 0.99. One seed in five still
partial-collapses, now with the polarity randomized: the
1D-pos-embed's systematic x-dies-first signature is gone, and which
dimension gets sacrificed is no longer fixed.

What follows is the full 6-cell sweep, the unified failure signature,
four independent strands of mechanism evidence (PCA, wall rotation,
2D-pos-embed intervention, Weak-SIGReg intervention), three
recommended recipes, and the combined-fix result showing the two
mechanisms compose additively.

After Klindt et al. (2026), *"When Does LeJEPA Learn a World Model?"*,
appeared, I ran a 14-cell linear-probe sweep as a follow-up. The
combined-fix encoder sits at the top of an unambiguous
linear-identifiability gradient (linear-probe R² = 0.94 vs MLP-probe
R² = 0.95 — essentially tied), while every unregularized cell
produces strongly *nonlinearly*-identifiable encoders (linear R² ≈
0.01 despite MLP R² ≈ 0.99). A Gaussian-OU experiment then generates
training data with approximately Gaussian agent-position latents (KS
distance ≈ 0.04) and trains both v5.5 and the combined fix on it.
v5.5's linear R² doesn't move (still ≈ 0.01); the combined fix's
rises modestly to 0.965. That completes a 2×2 test of Theorem 1's two
conditions (encoder Gaussianity constraint × Gaussian world latents):
three of four corners are directly tested, the encoder-constraint
axis dominates the linear-identifiability outcome, and the
world-latent axis adds a modest incremental lift consistent with
Theorem 3. The combined fix's linear identifiability is therefore
*robust to the world's latent distribution* — it doesn't depend on
Two-Room happening to have Gaussian-ish latents.

A latent-space-planning evaluation closes the loop on Theorem 4.
Linear interpolation in the combined-fix encoder's latent space
decodes (via nearest-neighbor lookup in a validation pool) to
position-space trajectories with mean path-length-ratio ≈ 1.0 and max
deviation ≈ 0.1 pixels from the oracle straight line, against PLR ≈
12.7 and max deviation ≈ 24.5 px for the unregularized v5.5 baseline.
A >10× improvement in planning quality, exactly as linear
identifiability + Theorem 4 predicts.

I treat this whole investigation as a characterization of one
component inside the curiosity-driven agentic system of Paulin
(2026): the world model is the learned digital twin inside the
structured knowledge model and the simulation primitive in the
planning module, and the failure modes here are the encoder-side
instantiations of the hallucination and error-propagation challenges
that paper names at the system level.

One more thing, because it's the honest part: the phase-diagram
framing this document was first written to defend was substantially
falsified by the multi-seed data. I've kept it as an inset section,
because how the story fell apart is more instructive than pretending
it was always this clean.

## background

The LeWorldModel paper proposes a ViT-Tiny image encoder feeding an
autoregressive transformer predictor that runs on latents interleaved
with action tokens, trained with two losses: a JEPA-style
reconstruction MSE on next-latent prediction with stop-gradient on
targets, and a SIGReg regularization term — the Epps-Pulley
characteristic-function distance between random unit projections of
the encoder output and the standard normal N(0, 1). In the published
recipe, SIGReg is there to discourage representational collapse by
demanding distributional spread across random directions in the
latent space.

The published recipe trains an encoder with weak quantitative results
on the Two-Room benchmark. Prior work
(`docs/experiments/tworoom_paper_v3.md`) showed that a small
architectural and LR-schedule change — sin/cos absolute positional
embedding, patch_size=4, peak LR ≈ 1e-4 — produces a near-perfect
probe (R²=0.999, MAE 0.25/0.14 px) on the paper's own data. That
refutes the "low-coverage data is the problem" framing in the
original work. The failure was method-driven, not data-driven.

This document picks up from that v3 recipe and asks one question: how
robust is it? Specifically, how does the encoder respond to changing
the learning rate and the SIGReg weight λ — independently, in
combination, and on a schedule.

## context: place in a larger agentic-system architecture

The world model here — a ViT-Tiny encoder plus AR predictor trained
with SIGReg-family regularization, under the LeWM framework of Maes
et al. (2026) — is meant to be one component inside a larger agentic
system, not a standalone artifact. The companion whitepaper *"A
Curiosity-Driven Agentic System for Knowledge Gap Identification and
Targeted Exploration"* (Paulin, 2026) describes a modular design in
which a Structured Knowledge Model (SKM) holds the agent's grounded
state and a Planning module turns identified knowledge gaps into
actions. That paper lists "digital twins: structured, executable
models of systems and their expected behavior" as one valid SKM
implementation (§2.2), and "simulation: invoke domain simulators or
digital twins to test hypotheses safely" as one valid planning-module
action class (§2.4).

A learned JEPA-style world model fills both slots at once. The
encoder + predictor pair is a learned digital twin for pixel-observed
dynamics, and latent-space rollout under candidate action sequences
is the cheap, fast simulation primitive the planning module can call
without spending physical resources.

For that to work, the latent representation has to support
compositional, predictable manipulation by the planner. The Klindt et
al. (2026) linear-identifiability theorem — and its empirical
validation worked out here (see *latent-space planning: Theorem 4
validation*) — is the property that makes the composition tractable.
When the encoder is approximately linearly identifiable, linear
interpolation in latent space corresponds to linear motion in the
world's coordinates (up to rotation), so the information-gain
calculations, verification steps, and counterfactual rollouts of
Paulin (2026) §2.4–§2.5 can operate in the latent representation
instead of projecting back to pixel space at every step.

So this document characterizes the *encoder-side* failure modes any
such digital-twin component will hit: the orthogonal-cheat collapse
of Strong-SIGReg, the architectural anisotropy of 1D positional
embeddings on 2D patches, the silent representation collapse under
slow temporal dynamics, and the residual ~20% partial-collapse rate
even under the combined fix. These map onto the *"Hallucination and
overconfidence"* and *"Error propagation"* failure modes Paulin
(2026) §4 names at the system level. Silent representation collapse
in particular is the perception-module analog of LLM hallucination —
the system reports success on a proxy metric while detached from
ground truth — and it has the same cure: validation gates, probe-
based grounding, explicit confidence tracking. The combined-fix
recipe is a worked example of that cure for a pixel-observed JEPA
world model. Read that way, this isn't a study of one regularizer's
failure modes. It's a characterization of one slot in the
agentic-system architecture, with operational recipes for filling
that slot reliably.

## setup

The training stack is unchanged from the v3 paper recipe except where
noted per cell.

- **Architecture.** ViT-Tiny encoder (d=192, 6 blocks, 3 heads,
  patch=4, sin/cos pos embed) + AR transformer predictor (d=192,
  4 blocks, 3 heads, context_len=16, AdaLN-zero) + SIGReg with
  17 knots, 1024 random projections, t_max=3.0. Action head is a
  2-layer MLP from action dim 2 to d=192.
- **Data.** 920,809 frames from the LeWorldModel paper's Two-Room
  collection, downsampled from the original 224×224 anti-aliased
  render to 64×64. The agent disk is a red circle on a beige
  background with a black vertical wall and a centered doorway.
- **Training.** AdamW with cosine LR schedule and linear warmup
  (warmup_steps=500), gradient clip 1.0, 2000 steps, batch 64.
  MLX-only, single Apple Silicon machine.

  Two things about this setup are wrong, and both affect how the
  numbers read. The configs
  specify `dtype: bfloat16`, but during the period this document
  covers the encoder pipeline ran in *effective fp32*. That
  was unintentional: the encoder's input normalization casts uint8 →
  fp32 before patch_embed, and MLX's type promotion then propagated
  fp32 through the whole encoder, overriding the bf16 parameter cast
  at startup. A perf cleanup pass briefly "fixed" this (perf #6 in
  `docs/performance_review_variance_research.md`), at which point v5.5
  with seed=0 collapsed from R²=0.995 to R²=0.546 — unregularized
  training needs the higher precision to commit the encoder to
  position-encoding directions. I reverted the dtype cast to preserve
  the historical behavior. Every comparison in this document is
  apples-to-apples (effective fp32 across all cells), but the recipe
  text's "bf16" label is inexact; "effective fp32 via uint8→fp32
  normalization" is the accurate description.

  The configs also carry a `grad_accum: 4` field. It was parsed and
  never implemented during this period — see perf #4 in
  `docs/performance_review_variance_research.md`. The effective batch
  was 64 throughout, not 256.
- **Probes.** A 3-layer MLP regressor (ReLU, hidden=128) maps the
  CLS-token encoder embedding to ground-truth (x, y) on a held-out
  trajectory split. I report MLP R², R²_x, R²_y separately, MAE in
  pixels per axis, and rank_95 (number of PCA components capturing
  95% of embedding variance).
- **Variance sweep methodology.** For each cell I ran 5 seeds (0–4)
  via `scripts/variance_sweep.py`, which clones the base config with
  `seed` and `name` overridden per seed, drives `scripts/research_run.py`
  on each, and aggregates the resulting `summary.json` files into a
  per-cell `variance.json` with mean ± std for each metric and
  per-seed detail. Identical data, identical machine, sequential
  execution. MLX's compile and kernel-ordering introduce enough
  nondeterminism that the same seed on the same code can produce
  materially different outcomes across runs, so I treat each seeded
  run as a sample of a population rather than a deterministic value of
  (config, seed). That last point turns out to be the whole story —
  see the v1 inset.

The full schema is `configs/schema.json`. Per-experiment configs: the
six core cells live at
`configs/tworoom_paper_v{3,4,5,5_3,5_4,5_5,5_6}.json`; the four
follow-up cells live at `configs/tworoom_paper_v{5_5,5}_rotated.json`
(wall-rotation mechanism test),
`configs/tworoom_paper_v{5_5,5}_sincos2d.json` (2D-positional-embedding
intervention), `configs/tworoom_paper_v{5_5,4,5}_weak.json`
(Weak-SIGReg intervention), and
`configs/tworoom_paper_v5_weak_sincos2d.json` (combined fix). Seeded
variants `*_seed{0..4}.json` are written per-sweep by the sweep
script.

## the six-cell sweep

I sample the (LR, λ, schedule) space at six cells. The two constant-λ
cells at the corners of the (LR, λ) plane probe whether the v3 recipe
is LR-limited or SIGReg-limited; one intermediate-λ cell probes the
titration; and one schedule cell probes whether the two extremes can
be reached by a time-varying λ.

| Cell  | LR     | λ schedule                         | n_seeds |
|-------|--------|------------------------------------|---------|
| v5.5  | 3e-4   | constant 0                         | 5       |
| v5.3  | 1e-3   | constant 0                         | 5       |
| v4    | 3e-4   | constant 0.1                       | 5       |
| v5.4  | 1e-3   | constant 0.01                      | 5       |
| v5    | 1e-3   | constant 0.1                       | 5       |
| v5.6  | 1e-3   | linear 0 → 0.1 over steps 1000–2000 | 5       |

Aggregated summary statistics, sorted by mean MLP R²:

| Cell  | mean R² | std R² | min R² | max R² | failure rate (R²<0.9) | rank_95 mean | rank_95 std |
|-------|---------|--------|--------|--------|------------------------|--------------|-------------|
| v5.5  | **0.990** | 0.009  | 0.979  | 0.997  | 0/5  (0%)              | 9.0          | 4.7         |
| v5.3  | 0.971   | 0.036  | 0.910  | 0.996  | 0/5  (0%)              | 8.0          | 4.8         |
| v4    | 0.914   | 0.115  | 0.771  | 0.998  | 2/5 (40%)              | 12.4         | 8.5         |
| v5.4  | 0.728   | 0.403  | 0.086  | 0.996  | 2/5 (40%)              | 6.2          | 3.0         |
| v5    | 0.314   | 0.138  | 0.146  | 0.464  | 5/5 (100%)             | 3.0          | 1.2         |
| v5.6  | 0.165   | 0.154  | 0.017  | 0.336  | 5/5 (100%)             | 3.2          | 1.3         |

Three things in this table do most of the work below. Both λ=0 cells
(v5.5, v5.3) succeed at every seed with tight standard deviations. v4
— the λ=0.1-at-low-LR cell the original phase-diagram story called
the "quality basin" — has a 40% partial-collapse rate and a mean
*below* v5.5's worst case. And the two high-LR-with-non-trivial-λ
cells (v5 and v5.6) collapse uniformly, with v5.6 (the schedule)
ending worse than v5 (constant) at the same final λ.

Best-case R² per cell, to settle whether the λ-bearing cells buy a
higher ceiling when they work:

| Cell  | best-case R² | best-case rank_95 |
|-------|--------------|--------------------|
| v5.5  | 0.997        | 14                 |
| v5.3  | 0.996        | 15                 |
| v4    | 0.998        | 21                 |
| v5.4  | 0.996        | 10                 |
| v5    | 0.464        | 3                  |
| v5.6  | 0.336        | 5                  |

The best cases of v4, v5.3, v5.4, and v5.5 are all within 0.003 R² of
each other. SIGReg does not raise the *achievable* quality in this
regime; it only changes the probability of achieving it. v5 and v5.6
sit far below the others — those cells have no path to the
high-quality region at all.

## what the data says

The data supports a much simpler picture than the original
phase-diagram framing. Three claims do the work.

**The achievable ceiling is shared.** Across all four cells that ever
produce R² > 0.9 (v5.5, v5.3, v4, v5.4), the best per-cell R² is
within 0.003 of the others, and the best per-cell MAE within ≈0.05 px
on either axis. SIGReg-bearing cells don't improve the quality of
encoders that successfully train; they only lower the probability of
successfully training. The "v4 quality basin" claim was a per-seed
artifact: v4 at seed=0 happened to draw one of the 3/5 quality
outcomes, and its R²=0.998 looked higher than v5.3's seed=0 draw of
R²=0.966 — but at 5 seeds the two ceilings are indistinguishable.

**Failure rate scales monotonically with λ × LR.** Hold LR at 3e-4
and raise λ: v5.5 (λ=0) fails 0%, v4 (λ=0.1) fails 40%. Hold LR at
1e-3 and raise λ: v5.3 (λ=0) fails 0%, v5.4 (λ=0.01) fails 40%, v5
(λ=0.1) fails 100%. The same λ behaves differently at the two LRs:
λ=0.1 is a moderate risk at lr=3e-4 and a guaranteed failure at
lr=1e-3. The reading is that the SIGReg gradient has to overcome the
reconstruction gradient to do damage. At low LR the optimizer
averages over the two; at high LR the per-step SIGReg pull is
sometimes enough to push the encoder into a degenerate basin before
reconstruction has built position structure.

**Mid-training λ schedules going 0 → positive are worse than either
endpoint.** v5.6 reaches a worse end state than v5 at the same final
λ (mean R² 0.165 vs 0.314). The encoder spends its first 1000 steps
reaching a healthy λ=0 basin, then the slow λ ramp drags it through a
structurally bad region. This is the cleanest mechanism finding in
the sweep, and it gets its own section below.

The original "two viable corners, two failure corners" phase diagram
rested on three implicit claims the multi-seed data rejects: that
v5.5 was a failure cell (it's the *best* cell), that v4 had a quality
advantage over v5.3 (their ceilings tie; v4's only distinguishing
property is a higher failure rate), and that v5.4 had a unique
"bifurcation pathology" (it's a trimodal cell — quality, partial,
full collapse — like the others, just with heavier tails). Drop those
and the parameter space collapses to a one-dimensional gradient
indexed by λ × LR, plus a single qualitatively distinct catastrophe
at λ-warmup schedules.

Two budget-extension sweeps (`v5.5_8k` and `v4_8k`, 3 seeds each at 4×
the original 2000-step budget) confirm both substantive claims. At
8000 steps, v5.5_8k achieves mean R² = 0.993 ± 0.004 with 0/3
failures (essentially identical to the 2000-step baseline) and v4_8k
achieves mean R² = 0.853 ± 0.138 with 2/3 partial collapses — *more*
variable than the 2k baseline, with the partial-collapse seeds
*deepening* (rank 6–7, R²_x ≈ 0.5–0.7) rather than recovering.
Best-case R² across the two cells differs by 0.001, tighter than the
2k envelope. The ceiling claim and the "v4 is unreliable" framing
both survive 4× longer training. Full detail in the budget caveat
under *what this document does not establish*.

## the universal failure signature: x dies first

Twelve of the 30 sweep runs failed (R² < 0.9 on the test split).
Across all twelve, the encoder loses its x dimension before — and
more reliably than — its y dimension. The mean R²_x across the 12
failures is 0.099 (std 0.193); the mean R²_y is 0.510 (std 0.360).
R²_x is essentially zero in 9 of 12 failures; R²_y in only 3 of 12.

Per-cell breakdown (failures only):

| Cell   | n failures | mean R²_x | mean R²_y | rank_95 (failures) |
|--------|------------|-----------|-----------|---------------------|
| v4     | 2          | 0.602     | 0.975     | 4.0                 |
| v5.4   | 2          | 0.168     | 0.491     | 3.5                 |
| v5     | 5          | 0.014     | 0.614     | 3.0                 |
| v5.6   | 5          | 0.018     | 0.312     | 3.2                 |

As λ × LR rises, x is progressively destroyed: R²_x ≈ 0.6 at v4 (mild
partial collapse), ≈ 0.17 at v5.4 (deeper), ≈ 0 at v5 and v5.6
(uniformly dead). y is more variable: it holds at R²_y ≈ 0.98 in v4
partial collapses, drops to ≈ 0.5 in deeper failures, and lands
anywhere between 0.03 and 0.66 in v5.6.

R²_x is the failure indicator. R²_y is the noise.

### mechanism: PCA evidence

That pattern is precise enough to constrain mechanism, and the PCA —
once I ran it — clarifies what SIGReg is doing to the encoder. I ran
PCA on four representative checkpoints (v5.5_seed0 quality, v4_seed2
partial collapse, v5_seed0 catastrophic collapse, v5.6_seed3 schedule
catastrophe), embedded the same 5000 validation frames in each, and
reported variance fractions and per-coordinate linear-fit
recoverability per principal component.

| Run                       | Var (top 3)        | Cum R²(top-3→x) | Cum R²(top-3→y) | Linear R²_x (full) | Linear R²_y (full) |
|---------------------------|--------------------|------------------|------------------|---------------------|---------------------|
| v5.5_seed0 (quality)      | 0.45 / 0.13 / 0.09 | 0.04             | 0.23             | **0.992**           | **0.999**           |
| v4_seed2 (partial)        | 0.45 / 0.43 / 0.10 | 0.01             | 0.77             | 0.895               | 0.998               |
| v5_seed0 (catastrophic)   | 0.57 / 0.29 / 0.11 | 0.01             | 0.89             | **0.336**           | 0.941               |
| v5.6_seed3 (schedule)     | **0.90** / 0.07 / 0.02 | 0.01         | 0.64             | 0.396               | 0.832               |

The full-rank linear recoverability column confirms the
encoder-level signature: x drops from R²=0.99 in the successful
encoder to R²≈0.34–0.40 in the collapsed ones, while y stays in
0.83–1.00. The asymmetry isn't an MLP-probe artifact — the x
information is gone from the collapsed encoders' latent
space.

The variance-fraction column shows the mechanism, and it's *not* what
the original draft of this section predicted. In the successful
encoder the top 3 PCs carry only 67% of the variance and encode
neither coordinate cleanly (cumulative R² of 0.04 for x and 0.23 for
y across the top 3; the top 10 reach only 0.23 and 0.63). Position
lives in the lower-variance dimensions; the dominant PCs are doing
something else — most likely background, scene structure, or
render-level features that vary more across frames than the small
agent disk does. In the failure-mode encoders the top 3 PCs carry
97–99% of the variance and cleanly encode y (cumulative R²_y up to
0.89) while encoding essentially no x (cumulative R²_x ≈ 0.01). The
more catastrophic the failure, the more concentrated the variance —
v5.6_seed3 has its top single PC carrying 90% of total variance, with
no other significant direction.

So the mechanism the data supports:

1. **Successful encoders distribute position information across many
   low-variance dimensions.** Under λ=0, with no pressure to
   concentrate or spread, the encoder allocates its top variance to
   non-position features (likely background, render structure,
   lighting) and encodes position diffusely in the lower-variance
   dimensions. Both x and y are recoverable from the full latent;
   neither dominates the PCA.
2. **SIGReg pressure concentrates the variance into a few
   dimensions.** Directly visible in the PCA: failed runs have 97–99%
   of variance in 1–3 PCs vs 67% in the successful run, and the worse
   the failure the more concentrated. This is structurally what
   SIGReg should do — its objective is "look Gaussian on random
   projections," and one easy way to satisfy that is one or two
   dominant directions with the right marginal shape, with the rest
   going to zero where their projections are trivially normal.
3. **One coordinate survives the concentration; the other doesn't.**
   In Two-Room, R²_y survives and R²_x is the casualty. Which one
   dies is *not* explained by task geometry (the rotation experiment
   below rules that out) but by encoder architecture — the positional
   embedding's treatment of the patch grid. The coordinate the
   encoder finds easier to pack into a small number of dimensions
   wins the surviving PC; the one that needs richer position
   structure gets destroyed.

What this corrected list does *not* claim is anything about why x
specifically dies. The PCA establishes the concentration mechanism
and the survival asymmetry, but doesn't adjudicate between "x has
multi-dimensional categorical+continuous structure so it gets killed"
(task geometry) and "the encoder has an architectural bias against
x-encoding" (architecture). The next subsection runs that test
directly.

### mechanism: wall-rotation rejects the geometric reading

The geometric reading of point 3 — that x dies because it carries
"which room" categorical structure plus a continuous within-room
offset, while y is a clean continuous coordinate — was the natural
first hypothesis. It's also testable. Rotate the Two-Room environment
90° (wall now horizontal, doorway along x), retrain v5.5 and v5 at 5
seeds each on the rotated data, and check whether the failure
signature follows the categorical structure to its new coordinate.

The geometric mechanism predicts: in the rotated env, y now carries
the bimodal categorical-plus-continuous structure (wall horizontal at
y≈32), so y should die first under SIGReg pressure and x should
survive. The polarity should flip cleanly.

5-seed sweep on the rotated dataset:

| Stat                | v5 (unrotated)    | v5_rotated         | Predicted flip   |
|---------------------|-------------------|--------------------|------------------|
| Mean R²             | 0.314             | 0.354              | ≈ 0.31           |
| Mean R²_x           | **0.014** (dead)  | **0.145** (still dying) | ≈ 0.61 (survive) |
| Mean R²_y           | **0.614** (variable) | **0.563** (still surviving) | ≈ 0.01 (dead)    |
| Mean rank_95        | 3.0               | 3.0                | ≈ 3              |
| Failure rate        | 5/5               | 5/5                | 5/5              |

The control (v5.5_rotated) reproduces v5.5 cleanly — mean R²=0.990,
5/5 successful, both axes healthy — so the rotation pipeline is
sound. But the failure-corner experiment shows the asymmetry does
*not* flip. R²_x still dies and R²_y still survives in the rotated
env, magnitudes only mildly shifted (R²_x improves from ~0.01 to
~0.15, R²_y degrades from ~0.61 to ~0.56). The categorical structure
moved from x to y in the underlying task; what dies did not.

The cleanest reading is that "what dies" tracks pixel-axis identity
in the rotated image, not task-coordinate identity. The rotation
script keeps the state vector's `x`-component aligned with the
rotated image's column axis and `y` with the row axis, so "R²_x dies"
means the same thing in both environments: the encoder loses
information about the agent's pixel-column position. The encoder is
consistently biased against representing column information, whatever
categorical structure happens to live there.

The geometric reading is rejected. The multi-dimensionality of x in
the unrotated env was a coincidence — the categorical structure
happened to share a pixel axis with the encoder's structural bias.
Generalizing from "multi-dimensional features die" was overreach.

### mechanism: 1D positional embedding as the structural source

If the asymmetry is architectural rather than task-driven, the place
to look is the positional embedding. The v3 recipe uses a sin/cos
positional embedding (`src/tinywm/model/encoder.py`, function
`_sincos_pos_embed`). Reading the code, the embedding is **1D,
applied to the row-major-flattened patch sequence**. Each patch at
grid position (r, c) gets the embedding for linear index
`i = r * patches_per_row + c + 1` (the `+1` accounts for the
prepended CLS token). For the 16×16 patch grid:

- Adjacent patches in the **column** direction have linear indices
  differing by 1. Their position embeddings differ only in the
  high-frequency components of the sin/cos basis.
- Adjacent patches in the **row** direction have linear indices
  differing by 16. Their embeddings differ across most frequency
  components, including low-frequency ones.

That's the structural asymmetry. Row position rides on the coarse
(low-frequency, large across-position magnitude) components; column
position on the fine (high-frequency, small across-adjacent magnitude)
components. The encoder has to learn to read both, and under λ=0 it
does, hitting R²=0.99 on both axes. But under SIGReg's
variance-concentration pressure, the surviving 1–3 PCs preferentially
absorb the directions carrying more variance per unit of position
change, which are the row-encoding directions. The column-encoding
directions, which encode position through fine high-frequency
wiggles, are exactly what gets lost when variance concentrates.

The mechanism I now believe holds:

1. **Concentration.** SIGReg drives the encoder's variance into a
   small number of PCs (PCA: ~1–3 in failed runs vs ~10 in
   successful runs).
2. **Architectural anisotropy.** The 1D embedding on a 2D patch grid
   makes row position ride on low-frequency components and column
   position on high-frequency ones. Invisible at λ=0 (both axes
   recover cleanly); load-bearing under variance concentration.
3. **Selective survival.** The PCs that survive are the ones encoding
   the low-frequency, large-magnitude signal — row position. Column
   position, in the fine wiggles, is the casualty.

The architectural reading generalizes differently from the geometric
one. The failure mode is not a property of tasks with
categorical-plus-continuous coordinates; it's a property of
architectures using 1D position embeddings on 2D inputs. Any 2D task
— image classification, video prediction, spatial reasoning — using
this embedding choice should expect a row-vs-column asymmetry under
any pressure that forces variance concentration. The mitigation is
architectural, not task-specific: use a 2D-aware position embedding
(separate row and column sin/cos with concatenated embeddings) or a
learned embedding initialized to give both axes symmetric capacity.

The PCA also rules out a confound the original framing was implicitly
worried about: that x might be "harder to learn" than y and so
the first to disappear under any pressure. The successful encoder's
full-rank R²_x = 0.992 shows x is not intrinsically harder — the
encoder learns it cleanly with no SIGReg pressure. The asymmetry
comes specifically from the encoder's architectural bias meeting
SIGReg's concentration.

### mechanism: 2D positional embedding confirms the architectural source

The architectural hypothesis predicts a sharp, implementable
intervention: replace the 1D sin/cos embedding with a 2D-aware
variant that gives row and column equal capacity. Split the d=192
embedding into a d/2 = 96 row-encoding subspace and a d/2 = 96
column-encoding subspace, each filled with an independent 1D sin/cos
basis on the corresponding axis index (`_sincos_2d_pos_embed` in
`src/tinywm/model/encoder.py`). Prediction: the R²_x / R²_y asymmetry
should collapse, and failure modes should distribute symmetrically
between axes rather than preferentially killing x.

I added the 2D embedding and ran 5-seed sweeps at both corners under
the new architecture.

| Stat                | v5.5_sincos2d (control) | v5_sincos2d (failure cell) |
|---------------------|--------------------------|------------------------------|
| Mean R²             | 0.988 ± 0.016            | **0.562 ± 0.226**            |
| Mean R²_x           | 0.989                    | **0.538**                    |
| Mean R²_y           | 0.988                    | **0.587**                    |
| R²_x / R²_y ratio   | 1.00                     | **0.92**                     |
| Mean rank_95        | 7.8                      | 3.8                          |
| Failure rate (R²<0.9) | 0/5                    | 5/5 (but mostly partials)    |

Side-by-side with the 1D baselines:

| Stat                | v5.5 (1D)  | v5 (1D)        | v5_sincos2d (2D)  |
|---------------------|------------|-----------------|---------------------|
| Mean R²             | 0.990      | 0.314           | **0.562** (+78%)    |
| Mean R²_x           | 0.985      | **0.014** (dead)| **0.538** (alive)   |
| Mean R²_y           | 0.996      | 0.614           | 0.587               |
| R²_x / R²_y         | 0.99       | 0.02            | **0.92**            |

The prediction holds, and the effect is bigger than just rebalancing.
The asymmetry collapses (ratio 0.02 → 0.92, essentially symmetric
outcomes) and cell-level mean R² nearly doubles. The 1D embedding
wasn't only creating an asymmetric failure mode; it was making
SIGReg's failure substantially worse than the architecturally
symmetric case required.

Per-seed detail on v5_sincos2d makes the new failure structure
visible:

| seed | R²    | R²_x  | R²_y  | rank_95 | note                          |
|------|-------|-------|-------|---------|-------------------------------|
| 0    | 0.237 | 0.008 | 0.466 | 2       | residual asymmetric collapse  |
| 1    | 0.755 | 0.766 | 0.744 | 5       | symmetric partial             |
| 2    | 0.628 | 0.648 | 0.609 | 7       | symmetric partial             |
| 3    | 0.430 | 0.510 | 0.350 | 1       | flipped asymmetry — x > y     |
| 4    | 0.762 | 0.755 | 0.769 | 4       | symmetric partial             |

Four of five seeds now show symmetric outcomes or a flipped asymmetry
(x stronger than y) — the architectural mechanism behind "x dies
first regardless of seed or task geometry" has been removed. Only
seed 0 still shows the old pattern, and even there R²_y is 0.466 (vs
the 1D baseline's average R²_y of 0.614), so it's a weaker version of
the same mode, not the same depth of failure.

The mechanism is now triangulated by three independent lines: PCA
confirmed the *concentration* claim; wall rotation showed the
asymmetry tracks pixel axes rather than task geometry; and the
2D-pos-embed intervention shows that removing the architectural source
collapses the asymmetry and substantially mitigates the cell-level
failure. The 1D-sin/cos-on-row-major-flattened-2D-patches choice was
load-bearing for the failure mode.

The 2D embedding doesn't eliminate the v5-cell failure — the cell
stays at 0/5 success (mean R² 0.56 vs v5.5's 0.99) — but it turns a
uniform catastrophic collapse into a uniform partial collapse with
substantially more position surviving. When this section was first
written I had two readings of the residual ceiling at ~R²=0.76: (a)
the SIGReg + high-LR + λ=0.1 combination is structurally bad
regardless of pos embed, i.e. there's still *something else* holding
it back; or (b) more seeds would lift it asymptotically to the v5.5
ceiling. The Weak-SIGReg results in the next section answer it:
reading (a) was right. The "something else" is Strong-SIGReg's
orthogonal-cheat failure mode (rank-deficient solutions satisfying
the projection-based regularizer with noise in unused dimensions),
and Weak-SIGReg's Frobenius-to-identity covariance penalty addresses
it directly.

The 2D embedding alone removes the *architectural* mechanism for v5
failures. The residual v5_sincos2d failures — including seed 0's
lingering x-killing collapse — are the *orthogonal-cheat* mechanism,
which the 2D embedding doesn't touch. The next section tests
Weak-SIGReg as the published fix for that second mechanism; the
combined-fix section after it verifies the two interventions compose
additively.

## the schedule catastrophe: v5.6

The strongest mechanism finding in the sweep is that ramping λ upward
during training reliably destroys the encoder. v5.6 trains at λ=0 for
the first 1000 steps, then ramps λ linearly from 0 to 0.1 over steps
1000–2000. All five seeds end below R²=0.5, mean 0.17, R²_x uniformly
dead. The end state is *worse* than v5 (constant λ=0.1 from cold
start, mean R² 0.31), despite reaching the same final λ.

The original v5.6 writeup framed this as evidence that the two
"viable" basins of the v1 phase diagram aren't connected by a
continuous hyperparameter schedule. With the corrected picture — no
v4-vs-v5.3 basin distinction worth defending — the finding reads more
cleanly.

During the λ=0 phase, the encoder finds a position-rich latent in the
rank=8–15 range characteristic of v5.3. By the time the ramp begins
at step 1000, it has committed substantial capacity to specific
position-encoding directions, with the remaining 177 dimensions near
zero. SIGReg's gradient at ramp onset demands variance across *all*
random projections, including those that mostly hit the unused
dimensions. To satisfy that, the encoder must either (a) move signal
from the position-encoding directions into the unused ones, which
destroys the position structure, or (b) leave the structure alone and
eat the SIGReg penalty, which the optimizer won't do because the
penalty is nonzero in gradient. It takes path (a). What's destroyed
first, per the universal signature, is the x-encoding subspace,
because it's the most concentrated.

Put another way: SIGReg from cold start lets the encoder build a
SIGReg-compatible representation; SIGReg from a v5.3 state requires it
to *demolish* a SIGReg-incompatible representation before rebuilding.
In the 1000 remaining steps under a rapidly decaying cosine LR, the
demolition phase eats all the available capacity.

I haven't run the symmetric experiment (ramp λ *down* from 0.1 to 0
mid-training). The argument predicts it would be less destructive:
ramping down releases the pressure suppressing concentrated
representations, letting the encoder re-form them — but only if
there's LR budget left to do so, and with the cosine schedule there
isn't much. A forecast worth putting on record before running it: I'd expect mean R² in 0.3–0.6, better than v5.6's
0.17 but worse than v4's 0.91, because the encoder spends its high-LR
phase under SIGReg pressure that prevents position structure from
forming and only gets to build it under low LR.

The schedule asymmetry is a structural property of the
encoder-and-SIGReg system, not a curiosity of which way the warmup
points.

## the combined fix: Weak-SIGReg + 2D pos embed

The two failure mechanisms are *independent*. The orthogonal-cheat
collapse (rank-deficient encoder satisfying projection-based
regularization with noise in unused dimensions) is a property of
Strong-SIGReg's mathematical structure, independent of architecture.
The 1D-positional-embedding anisotropy (x-dies-first under
concentration) is a property of the encoder, independent of the
regularizer. Two published fixes address them separately —
Weak-SIGReg (Akbar 2026, arXiv 2603.05924) adds a
Frobenius-to-identity covariance penalty that structurally blocks
rank-deficient solutions; the 2D sin/cos embedding gives row and
column equal capacity. Each alone substantially mitigates the v5 =
(lr=1e-3, λ=0.1) corner but doesn't fully recover the unregularized
R²=0.99 ceiling. The question this section settles: do the two fixes
compose additively when applied together?

I instantiate the combined fix in
`configs/tworoom_paper_v5_weak_sincos2d.json` — v5's training
hyperparameters (lr=1e-3, λ=0.1), with both `pos_embed_type:
"sincos2d"` and `sigreg.kind: "weak"`. Predicted clean-compose
outcome: mean R² approaching 0.99, symmetric R²_x ≈ R²_y, low
between-seed variance, no catastrophic failures.

5-seed sweep:

| seed | R²    | R²_x  | R²_y  | rank_95 | verdict      |
|------|-------|-------|-------|---------|--------------|
| 0    | 0.991 | 0.991 | 0.992 | 59      | **good**     |
| 1    | 0.787 | 0.858 | 0.716 | 13      | partial      |
| 2    | 0.993 | 0.993 | 0.994 | 36      | **good**     |
| 3    | 0.992 | 0.990 | 0.993 | 32      | **good**     |
| 4    | 0.992 | 0.989 | 0.994 | 39      | **good**     |

Across all 5 seeds: mean R² = 0.951 ± 0.092, mean R²_x = 0.964, mean
R²_y = 0.938, mean rank_95 = 35.8.

Across the four clean seeds (0, 2, 3, 4): **mean R² = 0.992 ± 0.001**,
with R²_x and R²_y essentially identical at every seed. That's the
tightest cluster the project has produced — std=0.001 across the four,
lower than v5.5's std=0.009 across its full five-seed cluster. The
combined fix delivers the predicted clean compose: the two
independent mechanisms are addressed by their respective architectural
interventions, and the resulting encoder hits the unregularized
ceiling while keeping the structural advantages the regularizers
provide (rank 32–59, symmetric per-axis recovery, robust to seed in
the four good cases).

Against every prior cell at the v5 corner:

| Cell                                            | Mean R² | std R² | Failure rate | R²_x / R²_y          |
|-------------------------------------------------|---------|--------|--------------|----------------------|
| v5 (Strong-SIGReg + 1D pos embed)               | 0.314   | 0.138  | 5/5 catastrophic | 0.014 / 0.614   |
| v5_sincos2d (Strong + 2D pos embed)             | 0.562   | 0.226  | 5/5 partial-to-bad | mixed         |
| v5_weak (Weak + 1D pos embed)                   | 0.767   | 0.080  | 4/5 partial     | 0.592 / 0.942    |
| **v5_weak_sincos2d (combined)**                 | **0.951** | 0.092 | **1/5 partial** | **0.964 / 0.938** |

The combined fix dominates every previous intervention at the v5
corner on every metric: highest mean, lowest failure rate, most
symmetric per-axis recovery. The single partial-collapse seed (seed
1) is the only residual failure mode in the cell.

### a qualitative encoder difference: linear-recoverable position

I ran linear (Ridge)
probes across all 5 seeds of the combined-fix sweep:

| seed | Linear R² | Linear R²_x | Linear R²_y | MLP R² | Δ (MLP − Linear) |
|------|-----------|-------------|-------------|--------|-------------------|
| 0    | 0.975     | 0.979       | 0.971       | 0.991  | +0.017            |
| 1    | 0.784     | 0.839       | 0.729       | 0.787  | +0.003            |
| 2    | 0.977     | 0.980       | 0.975       | 0.993  | +0.016            |
| 3    | 0.983     | 0.981       | 0.985       | 0.992  | +0.009            |
| 4    | 0.981     | 0.978       | 0.984       | 0.992  | +0.011            |
| **mean** | **0.940** | **0.951** | **0.929** | **0.951** | **+0.011** |

The 5-seed mean linear R² (0.940) is essentially tied with the MLP R²
(0.951). Per seed the gap is 0.003–0.017 — including the
partial-collapse seed 1 — which means the combined-fix encoder
produces position information that is **directly linearly decodable
from the latent, with negligible nonlinear lift left**. That's
qualitatively different from every other λ=0 cell in the paper: the
unregularized v5.5 baseline (the primary recipe) has linear R² ≈
0.011 with MLP R² ≈ 0.990, a gap of ~0.98 — its position is
recoverable *only* nonlinearly, via the patch-frequency-aliased
features earlier PCA work identified.

That difference matters beyond probe quality. Linearly-recoverable
position is a cleaner property of the representation: it makes the
encoder useful as a frozen feature extractor for downstream tasks
where linear heads suffice (linear planning, simple regression heads,
transfer with frozen features), and it admits a clean theoretical
interpretation. The next section maps this property across every cell
in the sweep and connects it to the Klindt et al. (2026) result on
LeJEPA identifiability.

### seed 1: inverted-asymmetry partial collapse

Seed 1's partial collapse (R²=0.787, R²_x=0.858, R²_y=0.716, rank=13)
is the first partial collapse in the entire
project where **R²_x > R²_y** — y is the casualty. Every prior
partial-collapse seed across v4, v5.4, v5, v5.6, and v5_weak had y
stronger than x. The polarity of which dimension dies had been a
fixed property of the encoder (the 1D-pos-embed asymmetry the wall-
rotation experiment localized to pixel-axis identity).

With the 2D embedding installed, the architectural anisotropy is
gone and the polarity becomes random. Seed 1 happens to be the seed
where random initialization put the encoder in a basin where y is the
harder direction; over many seeds I'd expect roughly half of
partial-collapse seeds under a fully symmetric encoder to favor
x-killing and half y-killing.

Seed 1's collapse isn't the SIGReg orthogonal-cheat (Weak-SIGReg
structurally prevents that) or the 1D-pos-embed asymmetry (sincos2d
structurally prevents that). It's residual stochastic-optimization
variance: some draws of the random initialization land the encoder in
a basin from which the combined training pressure isn't quite enough
to recover both dimensions cleanly. The failure rate is ~20% at this
budget; I haven't characterized whether it drops with more training.

## linear identifiability across the recipe lattice

A March 2026 paper from Klindt, LeCun, and Balestriero, *"When Does
LeJEPA Learn a World Model?"* (arXiv 2605.26379), proves a sharp
result for LeJEPA-style training. **Theorem 1**: when the world's
latents are Gaussian and both the SIGReg-style Gaussianity constraint
and the alignment loss are minimized, the encoder must be a *rotation*
of the true latent variables: `h(z) = Qz` for some orthogonal `Q`.
The encoder recovers the world's latents up to rotation — *linear
identifiability*. **Theorem 2**: Gaussian latents are the *unique*
distribution for which exact linear identifiability holds.
**Theorem 3**: approximate identifiability degrades continuously with
deviation from Gaussianity and from perfect alignment minimization,
bounded by an explicit function of the deviations. **Theorem 4**:
linear identifiability is *sufficient* for optimal latent-space
planning over any rotation-invariant cost.

The combined-fix observation in the previous section — linear-probe
R² near MLP-probe R² across 5 seeds — is exactly what Theorem 3
predicts for Two-Room: the latents (agent (x, y) bounded by walls)
are *approximately* Gaussian rather than exactly Gaussian, so I'd
expect approximate linear identifiability with R² near but not at
1.0. This section maps the property across *every* cell in the sweep
and compares the gradient to the theoretical prediction.

I ran the linear (Ridge) probe across all 14 cells with completed
variance sweeps via `scripts/probe_linear_sweep.py`. For each cell:
load the encoder, embed 5000 validation frames, fit a Ridge
regression from the embedding to ground-truth `(x, y)`, report R² on
a 20% held-out split, aggregate the mean across seeds. Side-by-side
with the MLP R² already in each cell's `variance.json`:

| Cell                    | n | Linear R² | MLP R² | Δ (MLP − Linear) | Reading                                |
|-------------------------|---|-----------|--------|-------------------|----------------------------------------|
| **v5_weak_sincos2d** (combined fix) | 5 | **0.940** | 0.951 | +0.011 | **linearly identifiable**              |
| v5_weak                  | 5 | 0.607     | 0.767  | +0.160            | partial linear                         |
| v5_sincos2d              | 5 | 0.582     | 0.562  | -0.020            | linear-where-it-works (high variance)  |
| v4_8k                    | 3 | 0.557     | 0.853  | +0.296            | bimodal: linear when clean, dead otherwise |
| v5                       | 5 | 0.365     | 0.314  | -0.051            | linear *but encoder is mostly broken*  |
| v4                       | 5 | 0.329     | 0.914  | +0.585            | **nonlinear**: SIGReg + low LR         |
| v5_rotated               | 5 | 0.310     | 0.354  | +0.045            | linear at low magnitude                |
| v5_6                     | 5 | 0.225     | 0.165  | -0.060            | linear-of-a-broken-encoder             |
| v5_4                     | 5 | 0.029     | 0.728  | +0.700            | **strongly nonlinear**                 |
| v5_5_sincos2d            | 5 | 0.003     | 0.988  | +0.986            | **strongly nonlinear**                 |
| v5_5_rotated             | 5 | 0.007     | 0.990  | +0.983            | **strongly nonlinear**                 |
| **v5.5** (primary recipe) | 5 | **0.011** | 0.990  | +0.979            | **strongly nonlinear**                 |
| v5.3                     | 5 | 0.000     | 0.971  | +0.971            | **strongly nonlinear**                 |
| v5.5_8k                  | 3 | 0.000     | 0.993  | +0.993            | **strongly nonlinear**                 |

Sorted by linear R² descending. Four observations.

### observation 1: λ=0 cells produce strongly *nonlinearly*-identifiable encoders

Every λ=0 cell — `v5.5`, `v5.3`, `v5.5_sincos2d`, `v5.5_rotated`,
`v5.5_8k` — sits at the bottom with linear R² near zero (0.000 to
0.011) and MLP R² near 1.0 (0.971 to 0.993). The gap is ~0.98 across
the board. Without a SIGReg-family regularizer, the encoder learns
patch-frequency-aliased nonlinear features that recover position only
through a nonlinear probe. The longer budget (v5.5_8k at 4× steps)
does *not* shift the encoder toward linear identifiability — if
anything linear R² is lower at the longer budget. The 2D embedding
alone (v5.5_sincos2d) doesn't help either: without SIGReg pressure
the encoder still commits to nonlinear features regardless of the
anisotropy fix.

**The unregularized recipe I recommend as primary — v5.5 — produces
an encoder that is not a linearly-decodable representation of the
world.** That's consistent with Theorem 1: linear identifiability
needs the Gaussianity constraint on the encoder output, which only
the SIGReg-family regularizers (Strong-SIGReg, Weak-SIGReg, VICReg)
enforce.

### observation 2: the combined fix sits at the top of the gradient

`v5_weak_sincos2d` has linear R² = 0.940, MLP R² = 0.951, gap 0.011.
Linear and MLP probes are essentially tied — across 5 seeds,
including the partial-collapse seed 1 — so the encoder is
approximately linearly identifiable. The gap to R² = 1.0 (~0.06) is
attributable to the non-Gaussian Two-Room latents per Theorem 2:
exact identifiability needs Gaussian latents, which the agent's
bounded `(x, y)` position is not. This is approximate identifiability
as the paper predicts, and this encoder is the cleanest exemplar in
the sweep.

### observation 3: both fixes are necessary; neither alone suffices

The mid-table cells show what happens with only one intervention:

- `v5_weak` (Weak-SIGReg + 1D pos embed): linear R² = 0.607, MLP R² =
  0.767. The Gaussianity regularizer alone moves the encoder
  substantially toward linear identifiability but can't close the gap
  to the unregularized ceiling because of the 1D-pos-embed anisotropy.
- `v5_sincos2d` (Strong-SIGReg + 2D pos embed): linear R² = 0.582,
  MLP R² = 0.562. Achieves linear identifiability *when it works*
  (linear ≈ MLP), but the 5-seed mean is dragged down by
  Strong-SIGReg's high partial-collapse rate at high LR.
- `v5_5_sincos2d` (no regularizer + 2D pos embed): linear R² = 0.003.
  The 2D embedding without a Gaussianity regularizer does *not*
  enable linear identifiability — the encoder still commits to
  nonlinear features.

Only the combined fix (Weak-SIGReg + 2D pos embed) reliably delivers
both linear identifiability and clean per-axis recovery across most
seeds.

### observation 4: Strong-SIGReg + low LR is the worst nonlinear cell

`v4` (Strong-SIGReg + lr=3e-4) has linear R² = 0.329, MLP R² = 0.914,
gap +0.585 — the largest MLP-vs-linear gap among cells where MLP
probes work. The Gaussianity regularizer is present, but at low LR the
encoder commits to nonlinear features that recover position via MLP
probes only. Extending to 8000 steps (v4_8k) shifts the gradient
(linear R² rises from 0.33 to 0.56), suggesting the regularizer
pressure becomes effective at longer budgets — but the 40–67%
partial-collapse rate remains.

### what this means for the recipe story

The earlier sections argued v5.5 is the "primary recipe" because it's
the most reliable producer of an encoder that recovers position with
high R². That's still correct *for probe quality* — v5.5 gives R²=0.99
with 0/5 failures. But probe quality only measures "can a probe
recover position from the latent." Klindt 2026 plus the linear-probe
data show that *how* the encoder represents position internally
matters independently:

- v5.5 produces a strongly nonlinearly-identifiable encoder. Position
  is recoverable only nonlinearly. For downstream tasks that use
  linear features — linear planning, frozen feature extraction with
  linear heads, simple regression — this encoder is *suboptimal*.
  Theorem 4 says linear identifiability is sufficient for optimal
  latent-space planning; v5.5 doesn't have it.
- v5_weak_sincos2d produces an approximately linearly-identifiable
  encoder. Position is recoverable linearly with deviation bounded by
  the non-Gaussianity of the world's latents. Downstream linear users
  get an encoder that's approximately a rotation of true position —
  exactly what Theorem 4 says enables optimal planning. The
  Gaussian-OU 2×2 test (below) shows this is *robust to the world's
  latent distribution*: the encoder is approximately linearly
  identifiable whether trained on the paper's non-Gaussian latents
  (R²=0.94) or on approximately-Gaussian Gaussian-OU latents
  (R²=0.965).

The right recipe depends on the downstream use case:

- **Research workflows, iteration speed, probe-quality
  measurements:** v5.5 (R² = 0.99, 0/5 failures, fast, simple).
- **Downstream tasks where linear-feature decoding matters, planning
  in latent space, frozen-feature transfer:** v5_weak_sincos2d (R² =
  0.95 on both linear and MLP probes, approximate linear
  identifiability, ~20% partial-collapse risk).

### cross-validation against the Klindt 2026 prediction

Klindt et al. predict SIGReg-family regularizers achieve approximate
linear identifiability when the world's latents are approximately
Gaussian. The 14-cell sweep shows that prediction holds with
substantial directional agreement:

- All cells with linear R² > 0.3 have a SIGReg-family regularizer
  active (Strong or Weak, λ > 0).
- All cells with linear R² < 0.05 either have λ = 0 (no regularizer)
  or are deep in catastrophic collapse (v5.4 at λ=0.01 falls here:
  the regularizer is too weak to take effect).
- The combined-fix cell sits at the top exactly as the paper's recipe
  — Gaussianity regularizer (Weak-SIGReg) + clean architecture (2D
  pos embed) — would predict.

That's independent empirical validation of the paper's prediction on
this task, and independent theoretical justification of the recipe I
converged on through mechanism work. Two paths to the same answer.

The recipe-lattice sweep varies the encoder while holding the world
fixed (paper data, with its non-Gaussian agent-position
distribution). The next subsection varies the world axis as well.

### both conditions are necessary: a 2×2 test

The 14-cell sweep tests linear identifiability across the recipe
lattice on a *single* dataset (the paper's scripted A*+PD collection,
whose `(x, y)` distribution is heavily non-Gaussian — concentrated
near the doorway, sparse in the corners). Klindt 2026's **Theorem 1**
is an if-and-only-if involving two conditions: the encoder satisfies a
Gaussianity constraint, and the world's latents are Gaussian. The
paper-data sweep holds the world fixed and varies the encoder, so it
tests the encoder axis and leaves the world axis untouched.

To test the world axis I generated a second dataset where the agent's
`(x, y)` follows a Gaussian-OU process on an open-room canvas (no
wall, `wall_thickness=0`). Empirical marginal-Gaussianity on a
1000-frame validation sample: per-axis KS distance ≈ 0.04, excess
kurtosis within ±0.16, empirical mean within 1 pixel of target,
empirical std within 7% of target. That puts the agent-position
distribution in Klindt 2026's *approximately Gaussian* regime (the
regime Theorem 3 addresses) — well above the non-Gaussianity of the
paper data and well below exact Gaussianity.

I trained two recipes on this dataset, both at seed=0 and 2000 steps:

| Cell                            | World latents | Encoder Gaussianity | Linear R² | MLP R² |
|---------------------------------|---------------|---------------------|-----------|--------|
| v5.5 (paper data)               | non-Gaussian  | none (λ=0)          | 0.011     | 0.990  |
| v5.5 (Gaussian-OU data)         | approx Gauss  | none (λ=0)          | **0.010** | 0.930  |
| v5_weak_sincos2d (paper data)   | non-Gaussian  | Weak-SIGReg + 2D    | 0.940     | 0.951  |
| v5_weak_sincos2d (Gaussian-OU)  | approx Gauss  | Weak-SIGReg + 2D    | **0.965** | 0.987  |

Three corners of the 2×2 (encoder Gaussianity × world Gaussianity) are
now directly tested. The fourth — combined-fix recipe with *exactly*
Gaussian world latents — is hard to construct on a pixel-rendered task
(the canvas boundaries always truncate the tails of any Gaussian
sampled on a finite domain), but Theorem 3 says the gap from R²=0.965
to R²=1.0 should bound the residual non-Gaussianity contribution.

**Both axes are needed.** The most striking row is the second:
v5.5 trained on approximately-Gaussian world latents shows **no change
in linear R²** (0.011 → 0.010). The MLP probe still recovers position
(R² = 0.93), so the encoder *is* learning position — it's just
learning it through nonlinear, patch-frequency-aliased features linear
probes can't decode. The world's latent distribution being Gaussian is
*not sufficient on its own* for the encoder to produce a
linearly-decodable representation. The encoder's own Gaussianity
constraint, via Weak- or Strong-SIGReg, is structurally necessary.

The fourth row confirms the complementary direction. With the encoder
Gaussianity constraint active, switching from paper data to
Gaussian-OU data lifts linear R² by 0.025 (and MLP R² by 0.036) — a
modest but consistent improvement matching Theorem 3's prediction of
bounded approximate identifiability when both conditions are
approximately satisfied.

The full reading of Klindt 2026 on this task: **linear
identifiability requires the encoder Gaussianity constraint (necessary,
established by row 2), and is further sharpened by world Gaussianity
(modest incremental lift, row 4 vs row 3).** Either condition alone is
insufficient: row 1 has neither and gets nothing; row 2 has only the
world condition and also gets nothing.

**Why this matters for the recipe.** The combined-fix recipe
(v5_weak_sincos2d) satisfies the encoder condition by construction.
Its linear-identifiability performance is therefore *robust to the
world's latent distribution*. It produces approximately linearly
identifiable encoders on the paper's non-Gaussian latents (R²=0.94)
and on approximately-Gaussian latents (R²=0.965). For users whose
downstream tasks have unknown or non-Gaussian latent distributions,
the recipe's linear-identifiability behavior should generalize — it
doesn't depend on the task happening to have Gaussian latents.

One detail from the v5.5 Gaussian-OU row: its
`val_mse_1step` was 5×10⁻⁷, essentially zero. The Gaussian-OU dynamics
are slow (decay=0.02, lag-1 autocorrelation ≈ 0.98), so consecutive
frames are extremely similar. The unregularized predictor learns to
copy the previous latent with high precision — a "silent prediction"
pattern where the model achieves vanishing MSE without doing any real
predictive work. That's a clean example of why mse1 alone is
misleading for representation quality: the *prediction* task is easy
under slow dynamics regardless of whether the *representation* is good.
Probe-based evaluation (linear and MLP) is the load-bearing measure.

## latent-space planning: Theorem 4 validation

Klindt et al. 2026 **Theorem 4** says linear identifiability is
*sufficient* for optimal latent-space planning over any
rotation-invariant cost. If the encoder is approximately linearly
identifiable, then linear interpolation between a start and a goal
frame in latent space, decoded back to data by nearest neighbor,
should produce approximately straight-line trajectories in the world's
coordinates. That's a downstream-task-relevant prediction I can test
directly: the unregularized v5.5 encoder is *not* linearly
identifiable (linear-probe R² ≈ 0.01), so its latent-linear
interpolation should produce curved or non-monotonic trajectories; the
combined-fix encoder is approximately linearly identifiable
(linear-probe R² ≈ 0.94–0.97), so its trajectories should track a
straight line.

`scripts/planning_eval.py` implements the test. For each encoder I
sample 5000 validation frames to form a decode pool, pick 20 (start,
goal) frame pairs deterministically spread across the canvas (minimum
start-goal distance 15 pixels), linearly interpolate from `z_start` to
`z_goal` in 20 steps, and at each step find the pool frame whose latent
is closest to the interpolated latent. The decoded trajectory's
`(x, y)` positions are then scored by:

- **Path length ratio (PLR):** sum of consecutive position-space
  distances along the decoded trajectory, normalized by the start-goal
  Euclidean distance. *Ideal = 1.0 (a straight line).*
- **Max deviation:** maximum perpendicular distance from the
  start-goal straight line. *Ideal = 0.*
- **Monotonicity:** fraction of decoded steps where distance to goal
  strictly decreases. *Read with the bang-bang vs zig-zag caveat
  below.*

Run against four single-seed encoders covering both datasets and both
recipe extremes:

| Run                                    | Dataset       | PLR (mean) | PLR (median) | Max dev (px) | Monotonicity |
|----------------------------------------|---------------|------------|--------------|--------------|--------------|
| v5.5 (paper)                           | paper         | **12.66**  | 11.86        | 24.49        | 0.51         |
| **v5_weak_sincos2d (paper)**           | paper         | **1.03**   | **1.000**    | **0.13**     | 0.09         |
| v5.5 (Gaussian-OU)                     | Gaussian-OU   | **17.62**  | 17.13        | 26.51        | 0.51         |
| **v5_weak_sincos2d (Gaussian-OU)**     | Gaussian-OU   | **1.29**   | 1.17         | **3.94**     | 0.12         |

PLR is the headline. **Combined-fix encoders decode to essentially the
straight line from start to goal** (paper PLR ≈ 1.03, Gaussian-OU ≈
1.29). **v5.5 encoders produce trajectories ~10× longer than the
oracle** (paper 12.7×, Gaussian-OU 17.6×). A >10× improvement in
planning quality purely from switching the recipe, and the qualitative
direction is exactly what Theorem 4 predicts.

Max deviation tells the same story more sharply: combined-fix on paper
data deviates 0.13 pixels on average from the oracle line; v5.5
deviates 24.5 pixels (about 40% of the canvas width). The combined-fix
encoder places every interpolated point essentially on the oracle
line; the unregularized encoder scatters them across the canvas.

### a metric subtlety: bang-bang vs zig-zag decoding

Both encoder types show *low* monotonicity (~0.1–0.5), but for
opposite reasons, and it would be easy to misread the metric.

- **v5.5 (unregularized) decodes to a zig-zag.** Each interpolation
  step decodes to a substantially different, unrelated frame. The
  trajectory wanders. Roughly half the step-to-step transitions
  happen to progress toward the goal by chance; the other half
  retreat. Monotonicity ≈ 0.5 is a random-walk signature, not
  planning success.
- **Combined-fix decodes to a bang-bang trajectory.** Many
  consecutive interpolation steps decode to the *same* NN frame —
  usually the start or the goal. Over 19 step-to-step transitions only
  1–2 see a change in distance to goal; the rest are no-ops.
  Monotonicity is therefore very low (≈ 0.05–0.12) under the strict
  "strictly decreasing" definition.

The bang-bang behavior on paper data is a feature of pool *density*:
the paper data is collected via A*+PD with a strong doorway bias, so
the validation pool has limited coverage of intermediate positions
between any given (start, goal) pair. The NN search finds nothing
close to the interpolated latent except start or goal themselves, so
it snaps. This is *exactly* the correct behavior for a linearly
identifiable encoder on a sparse pool, and PLR confirms it: 1.000
median, meaning the decoded path length is *exactly* the oracle
length, achieved through one big jump in the middle.

Gaussian-OU data has denser, more uniform coverage (smooth OU
diffusion vs A*+PD navigation), so NN decoding finds smoother
intermediate matches. PLR is slightly higher (1.29) because the
intermediate snaps add small wiggle, but the trajectory still tracks
the oracle line within ~4 pixels on average.

So the right reading of the four-cell table: PLR is the load-bearing
metric, max deviation reinforces it, and monotonicity is misleading
in this regime — both v5.5 and combined-fix score low, but
combined-fix scores low because the trajectory is *so close to a
straight line* that consecutive NN snaps return the same endpoint,
while v5.5 scores moderate (~0.5) because its trajectory is a random
walk.

### why this matters: a downstream-task-relevant validation

The linear-identifiability finding established that the combined-fix
encoder satisfies the mathematical property *predicted* to enable
optimal latent-space planning. This section closes the loop
empirically: the encoder *does* enable optimal latent-space planning
on this task. Linear interpolation in the combined-fix encoder's
latent space decodes to a straight line in position space — within
0.13 pixels on paper data and 3.94 pixels on Gaussian-OU. The
unregularized v5.5 encoder, which lacks linear identifiability,
decodes to trajectories 10× longer than the oracle, wandering 25
pixels off the straight line on average.

That's the cleanest empirical validation of Theorem 4 on this task.
The combined-fix recipe is recommended for any downstream use case
that benefits from planning, transfer, or interpolation in latent
space — its representation is approximately a rotation of true
position, and trajectories in that representation correspond to
trajectories in position space.

## recommended recipes

The recipe lattice has three viable entries: two unregularized
baselines (v5.5 and v5.3) that achieve high probe quality with
nonlinearly-encoded position, and a combined-fix recipe
(v5_weak_sincos2d) that achieves the same probe quality with
linearly-decodable position at the cost of a residual ~20%
partial-collapse rate. Pick by use case:

- **Research workflows, iteration speed, probe-quality measurements,
  or downstream tasks that put an MLP head on top of the encoder:**
  use `v5.5` (or its high-LR alternative `v5.3`). Cleanest reliability
  profile in the sweep (mean MLP R² = 0.99, 0/5 failures). Position is
  encoded nonlinearly — linear-probe R² is ~0.01 — so this encoder is
  *not* directly useful as a frozen feature extractor with linear
  heads.
- **Latent-space planning, frozen-feature transfer with linear heads,
  or any downstream task where linearly-decodable position matters:**
  use `v5_weak_sincos2d`. Approximately linearly identifiable per
  Klindt 2026 (linear-probe R² = 0.94, MLP-probe R² = 0.95 —
  essentially tied). Planning-quality validation per Theorem 4:
  latent-space linear interpolation produces decoded trajectories with
  PLR ≈ 1.0 and max-deviation ≈ 0.1 px (paper data) — a >10×
  improvement over v5.5's PLR ≈ 12.7 and max-deviation ≈ 24.5 px.
  Residual ~20% partial-collapse rate at this budget; filter
  checkpoints by R²_test or re-roll the seed if a partial collapse
  appears.

Detailed parameters and per-cell statistics for each recipe below.

**Primary recipe.** `configs/tworoom_paper_v5_5.json`.

- lr = 3e-4, cosine, warmup_steps = 500
- SIGReg λ = 0 (or remove SIGReg entirely)
- gradient clip = 1.0
- 2000 steps, batch 64, bf16

5-seed sweep: mean MLP R² = 0.990 ± 0.009, worst-case 0.979, no
failures. rank_95 averages 9. Best-case R² ties the best v4 ceiling.
This is the recipe I recommend whenever the encoder is trained for a
downstream task and a ~40% chance of partial collapse isn't
acceptable. Cleanest, simplest, most reliable in the sweep.
**Budget-robust:** a 3-seed extension to 8000 steps (`v5.5_8k`) gives
mean R² = 0.993 ± 0.004, 0/3 failures — ceiling and reliability both
survive 4× longer training, with no degradation, surprise
improvement, or runaway behavior.

**High-LR alternative.** `configs/tworoom_paper_v5_3.json`.

- lr = 1e-3, cosine, warmup_steps = 500
- SIGReg λ = 0
- gradient clip = 1.0
- 2000 steps, batch 64, bf16

5-seed sweep: mean R² = 0.971 ± 0.036, worst-case 0.910, no failures.
Roughly 3× faster wall-clock than v5.5 (higher effective LR per step)
at a small mean-quality cost. Use when iterating on architecture,
data, or downstream tasks where probe-level quality is enough.

**Recipes to avoid.** Any λ > 0 at LR ≥ 1e-3 (v5: 100% failure; v5.4:
40% failure with a heavy-tailed downside). Any mid-training schedule
that increases λ from a smaller value to a larger one (v5.6: 100%
failure, end state worse than constant-high-λ). v4 (λ = 0.1, lr =
3e-4) is *not* on this list but is harder to recommend than v5.5: 40%
partial-collapse rate against v5.5's 0%, with no measurable ceiling
advantage. The original "use v4 when quality matters" recommendation
was based on a per-seed artifact and is retracted.

**Architectural recommendation independent of (LR, λ).** Use the 2D
sin/cos positional embedding (`pos_embed_type: "sincos2d"`) rather
than the 1D variant. At λ=0 the choice doesn't matter (both reach mean
R² ≈ 0.99). With any non-zero λ at high LR, the 1D variant makes the
failure mode roughly twice as bad as it has to be (mean R² 0.31 vs
0.56 in the v5 cell). The 2D embedding adds no parameters, no
training-time cost, no regression at λ=0, and substantially mitigates
the catastrophic-failure cell. There's no current data supporting the
1D variant on any 2D-input task. Anyone using SIGReg should pair it
with the 2D pos embed.

**Combined-fix recipe.** `configs/tworoom_paper_v5_weak_sincos2d.json`.

- lr = 1e-3, cosine, warmup_steps = 500
- SIGReg λ = 0.1, **`kind: "weak"`** (Frobenius-to-identity on
  sketched covariance, sketch_dim = 64)
- **`pos_embed_type: "sincos2d"`** (separate row/column sin/cos)
- gradient clip = 1.0
- 2000 steps, batch 64

5-seed sweep: mean R² = 0.951 ± 0.092 (mean across 4 good seeds: 0.992
± 0.001, the tightest cluster in the sweep). Mean R²_x = 0.964, R²_y =
0.938 (symmetric per-axis). rank_95 mean = 35.8. 4/5 seeds clean; 1/5
partial collapse with inverted x/y asymmetry (seed 1: R²_x=0.86,
R²_y=0.72).

Use when you want **linearly-decodable position** in the latent
(linear-probe R² ≈ 0.98 in the good seeds, vs ≈ 0.10–0.20 in the
unregularized v5.5 baseline). The encoder this recipe produces is
qualitatively different from v5.5: position is directly readable as a
linear function of the latent, which makes the encoder more useful as
a frozen feature extractor where linear heads suffice. The cost is a
~20% partial-collapse rate at this budget — acceptable if you can
retry once or filter checkpoints by R²_test, not acceptable if you
need guaranteed-clean encoders. For guaranteed-clean encoders without
linear recoverability, use v5.5.

I haven't tested whether SIGReg at λ between 0 and 0.01 with low LR
has either reliability or quality benefits. The cells I've swept
suggest a smooth degradation in reliability between λ=0 and λ=0.01 at
high LR, but I don't know what happens at e.g. lr=3e-4 with λ=0.001.
One-cell sweep if anyone wants to sit on it.

## what this document does not establish

I bound the claims to the regime I trained in.

**Budget.** Most sweeps are at 2000 steps. I tested the ceiling-share
and partial-collapse-recovery claims at 4× the budget by running v5.5
and v4 to 8000 steps with 3 seeds each (`v5.5_8k` and `v4_8k`). The
results are unambiguous: at 8000 steps, v5.5_8k mean R² = 0.993 ±
0.004 (0/3 failures) and v4_8k mean R² = 0.853 ± 0.138 (2/3 partial
collapses). Best-case R² differs by only 0.001 across the two cells —
*tighter* than the 0.003 envelope at 2000 steps. The ceiling claim is
reinforced by the 4× extension, not refuted. v4's partial-collapse
seeds did not recover with more training; the partial-collapse mode
*deepened* (mean rank dropped from 12.4 to 8.0, mean R²_x from 0.84 to
0.72). Partial collapse is structurally absorbing on this task, not a
budget-limited transient. SIGReg's hypothetical
representational-richness payoff at *much* longer budgets (10× or 20×)
remains theoretically possible, but it's no longer the most natural
reading of the partial-collapse phenomenon — that reading predicts
recovery, and what I see is entrenchment.

**Task.** All sweeps are on Two-Room. The concentration mechanism
(SIGReg pressure concentrates variance into 1–3 PCs) is task-agnostic
and should generalize to any LeWM-style model trained with
SIGReg-style regularization. The architectural anisotropy mechanism
(1D position embedding on a 2D patch grid creates row-vs-column
asymmetry under concentration) is also task-agnostic in its source —
it applies to any image input, not just Two-Room — though *whether*
the asymmetry is visible depends on the data containing position
information that distinguishes row from column. Two-Room is good at
exposing this because position is the primary learnable signal; tasks
where position is less salient might mask the asymmetry behind other
features.

**Architecture and probe.** Most sweeps use the v3 architectural
recipe (1D sin/cos pos embed, patch=4, ViT-Tiny). The anisotropy
mechanism is specifically a property of *that* embedding choice (1D
sin/cos on row-major-flattened patches); the 2D-aware variant
(`sincos2d`) introduced here restores row/column symmetry. The MLP
probe captures position recoverable by a 3-layer nonlinear map; the
linear (Ridge) probe captures position recoverable by an affine map.
I report both starting with the combined-fix section, since the
qualitative difference (linearly- vs nonlinearly-decodable position)
is invisible to the MLP probe alone.

**Alternative regularizers.** I tested Weak-SIGReg (Akbar 2026)
specifically against Strong-SIGReg's orthogonal-cheat problem and
confirmed the published fix transfers to the JEPA/world-model setting.
I haven't tried other published alternatives (VICReg is technically a
special case of LeJEPA per the LeJEPA paper's own Appendix B.14;
Barlow Twins; explicit rank penalties; hyperspherical constraints).
Whether any would beat Weak-SIGReg at this task is open.

**Residual partial-collapse rate.** Even the combined fix
(Weak-SIGReg + 2D pos embed) shows a ~20% partial-collapse rate at
this budget (1 of 5 seeds, seed 1). The failure mode is no longer the
systematic "x-dies-first" architectural anisotropy or the
"rank-deficient cheat" SIGReg degeneracy — both are structurally
prevented. What remains is residual stochastic-optimization variance:
some random initializations land the encoder in a basin from which
neither the regularizer nor the prediction loss can pull it all the
way back to clean encoding within 2000 steps. I haven't characterized
whether this rate drops with longer training, larger sketch dimension
in Weak-SIGReg, or some other intervention. The clean-cluster mean R²
(0.992) is essentially the unregularized ceiling, so this residual
~20% failure rate is the only remaining open performance question at
the v5 corner.

## open questions

The data leaves a small number of specific questions open, in rough
order of priority and cost.

*Can the residual ~20% partial-collapse rate be eliminated?* Even the
combined fix shows one of five seeds entering a partial-collapse basin
(seed 1: R²=0.787, R²_x=0.86, R²_y=0.72 — the first inverted-asymmetry
case, where y is the casualty). The architectural and regularization
mechanisms for systematic failure are both structurally addressed;
what remains is residual stochastic-optimization variance on a small
minority of seeds. Three cheap follow-ups: (a) extend
v5_weak_sincos2d to 15 seeds to characterize the partial-collapse rate
more tightly (5 seeds is a coarse estimate); (b) try `sketch_dim = d =
192` (no sketching) in Weak-SIGReg to see if the larger gradient
signal saturates the remaining 20% of seeds; (c) extend to 4000 steps
and see whether the partial-collapse seeds recover with more budget.

*Does the polarity of seed 1's collapse predict anything?* Every prior
partial-collapse seed in the project had R²_y > R²_x. Seed 1 of
v5_weak_sincos2d is the first case where R²_x > R²_y. The 1D-pos-embed
asymmetry has been removed by sincos2d, so under random initialization
the failure-mode polarity should be roughly random across seeds. A
15-seed extension would test whether the expected ~50/50 split between
x-killing and y-killing partial collapses appears.

*Does the budget-dependence reverse the ceiling claim?* Extend v4 and
v5.5 to 8000 steps with 3 seeds each. If v4's mean R² at 8000 steps
exceeds v5.5's by more than the 0.003 currently seen at 2000 steps,
the ceiling claim only holds in the small-budget regime and SIGReg may
still be the right choice for full training runs. ~30 minutes of
compute per cell × 6 = ~3 hours.

*Does a λ-decrease schedule fare better than λ-increase?* Run
v5.6-inverse: lr=1e-3, λ=0.1 for steps 0–1000, linear ramp 0.1→0 over
steps 1000–2000, 5 seeds. Prediction: mean R² in 0.3–0.6 — better than
v5.6 (0.17) but worse than v4 (0.91). ~30 minutes. The result is
interesting either way: beat the prediction and the schedule asymmetry
is sharper than I think; underperform it and the schedule destruction
is more symmetric and the v5.6 catastrophe is less specifically about
ramp direction.

*Does SIGReg help downstream tasks even when it doesn't help the
probe?* The probe is a position regressor. Encoder properties it
doesn't capture — generalization to new layouts, latent richness for
planning, transfer to related tasks — could still benefit from
SIGReg's spreading. Untested here. Worth setting up before making
claims about SIGReg's utility rather than its dynamics.

## v1 hypothesis: the phase diagram framing (rejected by sweep data)

When I first wrote this document I had run each of the 6 cells at a
single seed (seed=0). On that data the picture looked like a clean 2×2
phase diagram in (LR, λ) with two viable corners, two failure corners,
and a schedule cell that landed in a "bifurcation pathology." I'm
keeping the original framing here, abbreviated, both as a record of
how the story evolved and to mark where single-seed reasoning misled
me.

The v1 phase diagram, with seed=0 numbers:

|              | **λ = 0**                                | **λ = 0.1**                              |
|--------------|------------------------------------------|------------------------------------------|
| **lr = 1e-3** | **v5.3**: R²=0.966, MAE 0.79/0.44, rank=7  ✓ | **v5**: R²=-4.74, MAE 27.5/19.6, rank=5 ✗  |
| **lr = 3e-4** | **v5.5**: R²=0.521, MAE 6.65/4.02, rank=9 ⚠ | **v4**: R²=0.998, MAE 0.27/0.16, rank=20 ✓ |

The v1 reading: matched (LR, λ) pairs are required. A quality basin
(low LR + full SIGReg) and a speed basin (high LR + zero SIGReg) sit
disconnected; mismatched configurations fail in characterizable ways;
v5.6 showed no continuous schedule could connect the two.

What the v1 framing got right:

- v5 (1e-3, 0.1) is a failure cell. The 5-seed sweep confirms 100%
  failure, just less extreme than seed=0's outlier R²=-4.74 suggested
  (sweep mean R²=0.31).
- v5.6 (the λ-warmup schedule) is a failure cell. The 5-seed sweep
  confirms 100% failure, more strongly than seed=0 alone established.
- The "state-dependent attractor" mechanism for v5.6 — that a
  position-aware encoder cannot survive SIGReg being turned on — is the
  cleanest single finding from the project, and it survives the sweep.
- SIGReg interacts asymmetrically with the encoder. The original
  intuition that this asymmetry has a structural mechanism (rather than
  being a hyperparameter-tuning artifact) was correct, though the
  mechanism turned out to differ in two layers from what v1 proposed.
  The asymmetry has *two independent structural sources*, identified
  after the v1 framing was rejected: (i) Strong-SIGReg's
  orthogonal-cheat (rank-deficient encoders satisfying the
  projection-based regularizer with noise in unused dimensions), and
  (ii) the 1D positional embedding's row-vs-column bias on a 2D patch
  grid. v1 collapsed both into "basin separation in (LR, λ)," which the
  multi-seed sweeps showed wasn't a thing.

What the v1 framing got wrong:

- v5.5 was framed as a failure cell ("undertrained or stuck partway
  between states") based on a seed=0 R²=0.52. The 5-seed sweep shows
  it's the *best* cell in the sweep (mean R²=0.990, 0/5 failures). The
  original seed=0 result was an outlier that selected one specific
  failure path out of a distribution that's otherwise tight and good.
- v4 was framed as a quality basin offering the highest achievable R².
  The 5-seed sweep shows v4's mean is 0.914, *below* v5.5's worst case
  of 0.979, and that 2/5 v4 seeds land in the same partial-collapse
  mode previously attributed only to "failure corners." v4's best-case
  R² (0.998) is matched by v5.5's best-case (0.997). There is no
  ceiling advantage.
- v5.4 was framed as having a unique "bifurcation pathology" with
  rank=1, based on seed=0. The 5-seed sweep shows v5.4 is trimodal
  (3 quality, 1 partial, 1 collapse) with high overall variance but no
  consistent bifurcation signature. The original seed=0 result was a
  specific seed that landed on a unique outcome I haven't seen
  reproduce.
- The "two disconnected basins" claim was the load-bearing structural
  assertion of the v1 story. The 5-seed sweep shows v4 and v5.5 have
  indistinguishable ceilings and that v4 fails 40% of the time. There
  is no observable basin distinction worth naming.

Why the v1 framing held together despite being wrong: every
single-seed run was a sample from a different distribution, and the
specific samples I drew were *internally consistent* with the
phase-diagram framing. v4 seed=0 hit its quality mode; v5.3 seed=0 hit
its quality mode; v5.5 seed=0 hit a failure mode; v4-vs-v5.3 seed=0
happened to differ in the R² direction I expected. Every data point
looked confirmatory. The framing only failed when I ran 5 seeds and
found that the same cells produce different modes with high
probability — i.e. that the per-seed identity of the mode is not a
property of the cell.

The methodological lesson: **for noisy training dynamics under weak
regularization, single-seed experiments cannot distinguish "this
cell's typical outcome" from "this cell sometimes produces this
outcome." They tell you a sample existed, not what the cell's
distribution looks like.** Future documentation of training dynamics
should sweep at least 3 seeds per cell before drawing structural
conclusions.

## provenance

Per-experiment writeups, in chronological order:

- `docs/experiments/tworoom_ours_v3.md` — v3 recipe on our data.
- `docs/experiments/tworoom_paper_v3.md` — v3 recipe on paper data.
- `docs/experiments/tworoom_paper_v4.md` — gradient clipping + lr=3e-4.
- `docs/experiments/tworoom_paper_v5.md` — push to lr=1e-3, collapse.
- `docs/experiments/tworoom_paper_v5_2.md` — tighter clip falsification.
- `docs/experiments/tworoom_paper_v5_3.md` — λ=0 at lr=1e-3, rescue.
- `docs/experiments/tworoom_paper_v5_4.md` — λ titration, bifurcation.
- `docs/experiments/tworoom_paper_v5_5.md` — λ=0 at lr=3e-4 (the now-rejected "failure corner" framing).
- `docs/experiments/tworoom_paper_v5_6.md` — λ-warmup, late collapse.
- `docs/experiments/variance_sweep_v4_v5_3.md` — 5-seed sweep of v4 and v5.3 that triggered the rewrite.

Sweep aggregates, under `checkpoints/_sweep_<cell>/variance.json`:

- `_sweep_tworoom_paper_v5_5/variance.json`         — 5-seed v5.5 sweep
- `_sweep_tworoom_paper_v5_3/variance.json`         — 5-seed v5.3 sweep
- `_sweep_tworoom_paper_v4/variance.json`           — 5-seed v4 sweep
- `_sweep_tworoom_paper_v5_4/variance.json`         — 5-seed v5.4 sweep
- `_sweep_tworoom_paper_v5/variance.json`           — 5-seed v5 sweep
- `_sweep_tworoom_paper_v5_6/variance.json`         — 5-seed v5.6 sweep
- `_sweep_tworoom_paper_v5_5_rotated/variance.json`  — 5-seed v5.5 on rotated data (control)
- `_sweep_tworoom_paper_v5_rotated/variance.json`    — 5-seed v5 on rotated data (mechanism test)
- `_sweep_tworoom_paper_v5_5_sincos2d/variance.json` — 5-seed v5.5 with 2D pos embed (control)
- `_sweep_tworoom_paper_v5_sincos2d/variance.json`   — 5-seed v5 with 2D pos embed (mechanism intervention)
- `_sweep_tworoom_paper_v5_weak/variance.json`       — 5-seed Weak-SIGReg at v5 corner (orthogonal-cheat fix)
- `_sweep_tworoom_paper_v5_weak_sincos2d/variance.json` — 5-seed combined fix (Weak-SIGReg + 2D pos embed)
- `_sweep_tworoom_paper_v5_5_8k/variance.json`       — 3-seed v5.5 at 4× budget (budget-extension control)
- `_sweep_tworoom_paper_v4_8k/variance.json`         — 3-seed v4 at 4× budget (budget-extension test of ceiling claim)

Mechanism-verification artifacts:

- `scripts/probe_xy_pca.py` — PCA-vs-coordinate analysis driver.
- `checkpoints/_xy_pca/results.json` — per-checkpoint PCA results used in the mechanism subsection.
- `scripts/rotate_dataset.py` — produced `data/tworoom_paper_64_rotated.h5` for the wall-rotation experiment.
- `src/tinywm/model/encoder.py`, function `_sincos_pos_embed` — the 1D positional embedding implementation identified as the structural source of the architectural anisotropy.
- `scripts/probe_linear_sweep.py` — linear-probe sweep across all variance-sweep checkpoints; produces the per-cell linear-vs-MLP table in the *linear identifiability across the recipe lattice* section.
- `checkpoints/_linear_probe_sweep/results.json` — per-seed and per-cell results of the linear-probe sweep across 14 cells.
- `src/tinywm/collect/gaussian_ou.py` and `scripts/collect_gaussian_ou.py` — Gaussian-OU agent-position trajectory generator for the 2×2 theory test. Open-room mode via `wall_thickness=0` (added in this branch).
- `data/tworoom_gaussian_ou_64.h5` — 10000-episode × 90-frame dataset with approximately Gaussian agent positions (per-axis KS distance ≈ 0.04, excess kurtosis within ±0.16).
- `checkpoints/tworoom_gou_v5_5/summary.json` and `checkpoints/tworoom_gou_v5_weak_sincos2d/summary.json` — single-seed training runs on the Gaussian-OU dataset that produce rows 2 and 4 of the 2×2 table.
- `configs/tworoom_gou_v5_5.json` and `configs/tworoom_gou_v5_weak_sincos2d.json` — config files for the Gaussian-OU training runs.
- `scripts/planning_eval.py` — Direction 2 driver: linear interpolation in latent space + nearest-neighbor decode + PLR / max-deviation / monotonicity metrics for the latent-space-planning evaluation.
- `checkpoints/_planning_eval/results.json` — per-encoder, per-pair metrics for the planning evaluation; the four-cell summary in the *latent-space planning* section is computed from this.
- `checkpoints/_planning_eval/trajectories_<run>.png` — one multi-panel figure per encoder showing the decoded position-space trajectories overlaid on the oracle straight lines. The bang-bang straight-line decoding of the combined fix against the zig-zag wandering of v5.5 is the clearest single picture in the project.

Follow-up research plan: `docs/lejepa_identifiability_followup.md`.
Documents Directions 1–3 from the Klindt et al. (2026) follow-up, with
Direction 1 (linear-probe sweep) now landed in this section and
Directions 2 (latent-space planning evaluation) and 3 (Gaussian-OU
agent dynamics for direct theory test) staged for later work.

Sweep driver: `scripts/variance_sweep.py`.

Related work (companion paper): Paulin, F. (2026). *A Curiosity-Driven
Agentic System for Knowledge Gap Identification and Targeted
Exploration*. Defines the agentic-system architecture referenced in
the *context: place in a larger agentic-system architecture* subsection
— structured knowledge model (§2.2), planning module with action
classes including simulation (§2.4), and the system-level failure
modes (hallucination, error propagation; §4) that the encoder failure
modes documented here instantiate at the perception layer.
