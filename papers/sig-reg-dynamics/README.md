# sig-reg dynamics

**[SIGReg in LeWM training: a risk-vs-ceiling story](sigreg_dynamics.md)** · empirical study · 2026

An empirical study of the SIGReg / learning-rate interaction in a ViT-Tiny + AR-transformer LeWM trained on the Two-Room benchmark. The spine is a 5-seed variance sweep across six (LR, λ, schedule) cells, extended with four mechanism-intervention sweeps — wall rotation, 2D positional embedding, Weak-SIGReg, and the combined fix. Twelve cells, ~60 training runs.

The short version: SIGReg doesn't raise the encoder-quality ceiling at this budget, it adds failure risk that climbs with λ and learning rate. Every non-trivial failure shares one signature — the x dimension dies before y — produced by two interacting mechanisms: SIGReg's variance concentration and an architectural anisotropy from a 1D positional embedding sitting on a 2D patch grid. The combined fix (Weak-SIGReg + 2D-aware embedding) addresses both and yields linearly-decodable position (linear-probe R² ≈ 0.98), a property the unregularized baseline does not have.

No PDF export yet.
