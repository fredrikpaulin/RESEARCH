# Manufacture the data, generalize the model, ground the agent

**A position paper on building industry-general business competence from simulated data, and spending it on one real brick-and-mortar business at a time.**

Fredrik Paulin · June 2026 · position paper / preprint draft

---

## Abstract

Most of the economy is a long tail of small, physical businesses that are individually data-poor and collectively data-rich in the only sense that matters: they share mechanism. A bakery, a butcher, a hardware store, and a garden centre differ in inventory but obey the same handful of dynamics — perishability, lead times, labour constraints, seasonal demand, and the accounting identities that bind them. This paper argues a three-part position. First, that for an industry with strong mechanistic structure, a calibratable simulation can be made *accurate enough* to generate large volumes of grounded operational data — data that obeys conservation laws rather than a generator's imagination, and that includes the uncensored ground truth real businesses never record. Second, that this data, randomized across the industry's parameter space and combined with available public data, can instill a *world model* of the industry that generalizes across its members. Third, that real value reaches a *specific* business not through the world model alone but through an agentic harness that grounds the general model in that business's own sparse, lagged, censored data and acts under human oversight. The simulation bridges the data gap; the harness bridges the gap from general competence to specific value. We state the load-bearing claim — "accurate enough" — as a falsifiable, measurable property, situate the approach against world models, sim-to-real transfer, synthetic data, operations research, and agentic LLMs, and describe how it could be evaluated and how it could fail.

---

## 1. The problem: a data-poor long tail meets a data-hungry method

The frontier of machine learning is running into the edge of its fuel supply. Public human-generated text adequate for training large models is projected to be effectively exhausted somewhere between 2026 and 2032 at current scaling rates (Villalobos et al., 2024). The field's response has been to manufacture data — synthetic generation is now a mainstream strategy for scaling training (see the 2024 ACL survey on LLM-driven synthetic data generation, curation, and evaluation). The obvious hazard is that a model trained on the recursive output of models degrades, a failure mode now documented well enough to have a name: model collapse (Shumailov et al., 2024). Synthetic data is only as good as the process that grounds it.

Step away from the frontier and the data problem inverts. The typical brick-and-mortar small business is not short of *web* data; it is short of *its own* data. A bakery has a point-of-sale system, a supplier or two, a payroll spreadsheet, and an owner who already works more hours than is reasonable. It has no data engineering, no historian database, and no appetite for either. What records exist arrive late and unevenly — sales batched at close, deliveries logged hours after the van leaves, production counts entered when someone has a hand free. Whatever the data wall means for a frontier lab, the local bakery hit its own wall years ago and learned to live behind it.

And yet these businesses are not idiosyncratic. They are near-copies of each other. The structure of a perishable-goods retailer is not a trade secret; it is close to a physical law. Stock decays on a known clock. Demand follows day-part, weekday, season, and weather. Labour is constrained by competence and regulation. Money obeys double-entry whether anyone is watching or not. The individual business is data-poor, but the *industry* is structure-rich, and structure is a prior you can encode.

That asymmetry — sparse local data, dense shared mechanism — is the opening this paper is about.

## 2. The position

We hold that the right way to serve the data-poor long tail is to manufacture the missing data from mechanism, learn a transferable model of the industry from it, and deliver value to a single business through an agent that grounds that model in the business's own thin data. Concretely, three claims:

1. **Simulation can be made accurate enough to generate useful industry data.** For a vertical with strong mechanistic priors, a calibratable, parameter-randomized simulation produces operational trajectories that are grounded in conservation laws — and that include the uncensored ground truth (true demand, including the customers who left) that no real point-of-sale system records.

2. **A world model trained on that data generalizes across the industry.** Pretraining on simulated trajectories spanning the industry's parameter space, combined with available public data, yields a model of business *dynamics* — not facts about one shop, but competence about how shops of this kind behave — that transfers to members it has never seen.

3. **Value is delivered by an agentic harness, not by the model alone.** A specific business gets value when an agent grounds the general world model in that business's own sparse, lagged, censored data and proposes actions under a human approval throttle. The harness is the product; the model is the engine.

The crux is the phrase "accurate enough," and Section 5 treats it as the falsifiable claim it is. The rest is architecture and evidence.

## 3. Related work: this is mostly assembled from solved problems

The position is novel as a synthesis, not as a set of ingredients. Each ingredient has a literature, and the honest move is to name them.

**World models.** The idea that an agent can learn a compressed, predictive model of its environment and then learn to act *inside* that model is not new. Ha and Schmidhuber (2018) trained an agent largely within a generative model of its environment and transferred the resulting policy back to the real task. Hafner et al. (2020) made learning-in-imagination practical for control, propagating value gradients through trajectories dreamed in a learned latent space; the line now solves diverse control tasks from a learned world model alone (Hafner et al., 2025). We borrow the framing — learn the dynamics, then act against them — and apply it to a business rather than a video game or a robot.

**Sim-to-real and the reality gap.** Robotics learned to cross the gap between simulation and hardware not by making the simulator photorealistic but by making it *varied*. Domain randomization trains across a wide distribution of simulated conditions so that reality looks like one more variation; Tobin et al. (2017) transferred a detector trained only on non-realistic randomized renders to the real world. The lesson is precise and directly reusable: you do not need a perfect twin, you need enough variation that the real instance falls inside the training distribution. Our industry-parameter randomization is domain randomization wearing an apron.

**Synthetic data and its failure mode.** Synthetic data is now central to training (2024 ACL survey), and its central risk is recursive degradation — model collapse — when models train on their own undisciplined output (Shumailov et al., 2024). This is the strongest argument *for* a mechanistic generator over an LLM-as-generator: a simulation grounded in mass balance and double-entry does not collapse, because its outputs are constrained by conservation, not by a previous model's distribution.

**Digital twins.** Industry already builds high-fidelity virtual replicas of physical systems, synchronized to a specific instance and used as "what-if" testbeds (Grieves & Vickers, 2017; ISO 23247-1:2021). Our proposal is deliberately *not* a digital twin. A twin mirrors one machine in real time; we want a generative model of an *archetype* used to instil transferable competence, and the per-business artifact is an agent, not a mirror. The unit of reuse is the industry, not the instance.

**Agent-based modelling.** Economists have long argued for building systems bottom-up from interacting agents to capture emergent behaviour that equilibrium models miss (Farmer & Foley, 2009). Our high-fidelity simulation mode — individual customers with reasons to visit, individual staff with skills and fatigue — is agent-based modelling pointed at a single firm rather than an economy.

**Operations research.** The decision problems we want the agent to make are old and well-posed. The newsvendor model is the canonical treatment of how much of a perishable good to stock under uncertain demand, where unsold units are worthless at the end of the day — true of newspapers, of croissants, and eventually of everyone (Petruzzi & Dada, 1999). Critically, OR also formalizes the data pathology at the heart of retail: sales are a *censored* observation of demand, because a shelf that empties at nine records the sale and forgets the demand it could not meet. Estimating true demand under censoring is a known, hard problem, with recent datasets built specifically to study it in fresh retail (FreshRetailNet-50K, 2025; data-driven censored newsvendor, 2024). We return to this because the simulation has an unfair advantage here.

**Agentic LLMs, and one cautionary tale.** Reasoning-and-acting agents that interleave thought with tool use are now a standard pattern (Yao et al., 2023). The cautionary data point is Anthropic and Andon Labs' Project Vend (2025), in which a capable model was given a real micro-business — an office shop — to run. It was talked into discounts, priced goods below cost, and at one point hallucinated a supplier and took offence when corrected. The model was not the problem; the *absence of a harness* was. Project Vend is the single most useful piece of prior art for this paper, because it demonstrates exactly the failure the harness exists to prevent.

## 4. Architecture

The approach has three components, in the order you would build them.

### 4.1 The simulation as a grounded, generative world

The generator is a mechanistic simulation of the industry archetype, not a learned one. It advances an append-only event log; state is a projection of that log. Events are produced by a set of generative processes — demand, supply, production, spoilage, labour, and the occasional shock — each carrying domain mechanism explicitly rather than as a fitted curve. Stock is modelled as individual lots with expiries and states, not a scalar per product, because perishability and traceability demand it. Money is integer minor units under double-entry, so that nothing which has to reconcile is ever a float.

Two properties make the data worth training on.

*It is conserved, not imagined.* Because the simulation enforces mass balance on inventory and balanced double-entry on money, its trajectories cannot drift into the incoherence that produces model collapse. A simulated quarter is wrong only in the ways the model is wrong; it is never internally contradictory.

*It is randomized across the industry.* Following the domain-randomization lesson, the simulation's parameters — footfall, product mix, margins, lead times, staffing, seasonality — are sampled across the range an industry spans, not tuned to one shop. The training distribution is the industry, so a real shop is one more sample from it.

There is a fidelity dial. An aggregate mode treats demand as a probabilistic flow and runs whole simulated quarters in milliseconds, which makes randomized, many-seed generation cheap. A high-fidelity mode resolves the same world into individual customers and staff — agent-based modelling for the cases where emergent micro-behaviour matters. Both emit the same event vocabulary, so a model or policy trained against one faces no surprises from the other.

### 4.2 From simulation to world model

The world model is trained on simulated trajectories spanning the randomized industry, supplemented with available public data — calendars, weather, seasonal patterns, published price levels, the generic domain knowledge a large model already carries. The target is not a database of any one shop's facts; it is competence about the *dynamics* of shops of this kind: how demand for this category moves, how stock decays, how a roster fails, how a missed delivery propagates.

This is where the simulation earns its keep beyond mere volume. Real operational data is censored: a point-of-sale log records what sold, never what was wanted, and a product that sells out hides the demand a forecaster most needs to see. The censored-demand literature exists because recovering true demand from sales is hard (FreshRetailNet-50K, 2025). The simulation does not have to recover it — it *knows* it, because it generated both the demand and the sale. Training on simulated data means training on uncensored ground truth — a structural advantage no amount of real data can confer.

### 4.3 The agentic harness for one specific business

The general world model is then attached to a single business through an agent that grounds it in that business's own data and specifics: its recipes, its staff and their competencies, its suppliers, its actual (sparse, lagged) sales. The harness follows a strict discipline: agents *propose* actions, and a single executor — the only component permitted to change state — validates each proposal against schema, referential integrity, business rules, and accounting balance before anything is committed. A blast-radius approval tier acts as a throttle: small, reversible actions auto-apply; large or irreversible ones queue for a human. Early on the threshold sits low and a person signs off on nearly everything; as the system earns trust it rises. This is the scaffolding Project Vend lacked.

Two further design points matter for a real shop. The agent's view obeys an *observation contract*: it sees only what a real manager would — what the till and the sensors report — never the privileged ground truth the simulation used to train it. And because the data arrives late and unevenly, every event is stamped with both when it occurred and when it was learned, so the agent can reason about how stale its picture is before it acts. The lag is not engineered away; it is made visible and bounded, which also means the high-value work skews toward the batch and administrative tasks that tolerate it, rather than second-by-second operational control that does not.

### 4.4 Evaluation

The claim is testable, and a position paper should say how. Four levels:

1. **In-simulation.** Score a grounded agent against deterministic baselines and a do-nothing floor on a single economic objective — operating profit net of waste and labour, say — across many randomized seeds. Necessary, not sufficient.
2. **Sim-to-sim transfer.** Train the world model on one region of the industry's parameter space; test the grounded agent on held-out regions. This measures generalization directly.
3. **Sim-to-real.** Ground the model on a real business's history and compare its decisions against what actually happened, with particular attention to stockout-prone items where the de-censoring advantage should show.
4. **Shadow mode.** Run the harness alongside a real operation, proposing but not acting, and compare proposed outcomes to realized ones before raising the approval threshold. Cheap, reversible, and honest.

## 5. The load-bearing claim: what "accurate enough" means

The entire position rests on two words, so they deserve a definition that can be proven wrong.

"Accurate enough" does **not** mean the simulation reproduces a specific business's numbers. It means the simulation belongs to the same *equivalence class of dynamics* as the real industry — close enough that a world model or policy learned in simulation transfers its decision quality to a real member of the industry. The metric is not fidelity of any single quantity; it is transfer of *decision quality*. A simulation is accurate enough when an agent trained against it makes better decisions on a real business than the alternatives, and not before.

This reframing is what makes the bet tractable, and it is borrowed wholesale from robotics. Domain randomization worked not because the simulator was right but because it was varied enough that reality fell inside its span (Tobin et al., 2017). The same wager applies here, with one favourable difference: a bakery's mechanism is far more constrained than a robot's contact dynamics. Perishability, lead times, and accounting are close to deterministic; the irreducible uncertainty is concentrated in demand. That suggests where to spend fidelity — model the mechanism faithfully, randomize the demand widely — and where the reality gap will bite hardest.

It will bite. The simulation cannot anticipate mechanism it does not contain: a charismatic competitor opening next door, a viral product, a road closure. This is a real limit, and it is the reason the harness defers to a human at the top of the blast-radius scale rather than pretending to omniscience. The world model is a strong prior, not an oracle, and a system that confuses the two is a Project Vend waiting to happen.

## 6. Why this might work, and why it might not

The case *for*: the long tail's defining feature — shared mechanism — is the feature that makes a mechanistic simulation a good prior and a good data source. The data wall is a frontier problem; for a vertical with strong priors it is a non-problem, because the data can be manufactured from the mechanism without recursive collapse. The censored-demand pathology that hobbles real forecasting is sidestepped by construction. And the parts are individually proven — world models, sim-to-real transfer, agentic execution, the operations research — so the risk lives in the synthesis, not the components.

The case *against*, stated as plainly as the case for:

- **The reality gap may be wider than the mechanism suggests.** If demand in a real vertical is driven by factors the simulation cannot see, no amount of randomization rescues it. This is measurable in sim-to-real testing, and it may fail outright for some verticals.
- **"Accurate enough" may be expensive.** The fidelity required for transfer is unknown until measured, and it may be higher than is cheap to build or calibrate.
- **Generalization across an industry assumes the industry is homogeneous in mechanism.** Some "industries" are several different businesses wearing one label, and the world model will fragment along those seams.
- **Grounding on sparse local data may underdetermine the specific business.** A strong general prior plus too little local signal can yield confident, plausible, wrong advice — the most dangerous failure mode, and the reason human oversight is non-negotiable until calibration is demonstrated.

The falsifiable predictions are clean. If the position is right, a world model pretrained on randomized simulated industry data, grounded on a real business's thin logs, should beat both a generic large model with no world model and classical per-business baselines on that business's held-out decisions — and should beat them by the most on the stockout-prone, censored items where its uncensored training gives it an edge. If it does not, the position is wrong, and the experiment will say so.

## 7. Discussion and conclusion

The prevailing instinct when a business lacks data is to wait until it accumulates some. For the long tail that wait is effectively infinite, because the business will never generate frontier-scale data about itself and would not know what to do with it if it did. The argument here is that you do not need the business's data to build competence about the business's *kind*. You need its mechanism, which is shared, and which you can simulate; and then you need a thin thread of the business's own data to ground that competence into advice, which an agent can hold under supervision.

If this holds, the unit of reuse is the industry archetype: simulate it once, well enough, and every member of the industry becomes a grounding problem rather than a training problem. The data wall does not apply to data you manufacture from a conserved mechanism, and the censoring that cripples real forecasting does not apply to data whose ground truth you generated yourself. That is a comfortable position to argue from. Whether it is a correct one is an empirical question, and the honest end of a position paper is to say that out loud, name the measurements that would settle it, and go and take them.

The frontier is running out of the internet. The long tail never had it. The same tool — disciplined synthetic data from a mechanism you actually understand — might answer both, and the smaller problem is the one where it can be checked against reality by close of business.

## Companion paper

The argument is only half the work; the machinery is the other half. Its specification lives in the companion technical whitepaper, *From simulation to a vertical world model* (`whitepaper-sim-to-worldmodel-pipeline.md`), which details the four-stage pipeline this paper assumes: the parameter-randomized, conservation-constrained generator and its uncensored labels; the event-trajectory corpus and its tokenization; the world-model architecture, training objectives, and per-business grounding; and the evaluation protocol — built around held-out parameter regions and censored-demand recovery — that would confirm or refute the thesis above. An alternative route — predicting in latent space rather than generating events — is specified in the joint-embedding companion, *Predicting in latent space* (`whitepaper-joint-embedding-pipeline.md`). This paper is the *why*; those two are the *how*.

---

## References

- Anthropic & Andon Labs (2025). *Project Vend: Can Claude run a small shop? (And why does that matter?)* https://www.anthropic.com/research/project-vend-1
- Farmer, J. D., & Foley, D. (2009). The economy needs agent-based modelling. *Nature*, 460, 685–686. https://www.nature.com/articles/460685a
- FreshRetailNet-50K (2025). *A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail.* arXiv:2505.16319. https://arxiv.org/abs/2505.16319
- Grieves, M., & Vickers, J. (2017). Digital Twin: Mitigating Unpredictable, Undesirable Emergent Behavior in Complex Systems. In *Transdisciplinary Perspectives on Complex Systems*. (See also ISO 23247-1:2021, Digital twin framework for manufacturing.)
- Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. *NeurIPS 2018*. arXiv:1803.10122. https://arxiv.org/abs/1803.10122
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR 2020*. arXiv:1912.01603. https://arxiv.org/abs/1912.01603
- Hafner, D., et al. (2025). Mastering diverse control tasks through world models. *Nature*. https://www.nature.com/articles/s41586-025-08744-2
- Petruzzi, N. C., & Dada, M. (1999). Pricing and the Newsvendor Problem: A Review with Extensions. *Operations Research*, 47(2), 183–194. https://pubsonline.informs.org/doi/10.1287/opre.47.2.183
- Shumailov, I., et al. (2024). AI models collapse when trained on recursively generated data. *Nature*, 631, 755–759. https://www.nature.com/articles/s41586-024-07566-y
- *On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey* (2024). *Findings of ACL 2024.* arXiv:2406.15126. https://arxiv.org/abs/2406.15126
- *The Data-Driven Censored Newsvendor Problem* (2024). arXiv:2412.01763. https://arxiv.org/abs/2412.01763
- Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. *IEEE/RSJ IROS 2017*. arXiv:1703.06907. https://arxiv.org/abs/1703.06907
- Villalobos, P., et al. (2022, rev. 2024). Will we run out of data? Limits of LLM scaling based on human-generated data. *Epoch AI.* arXiv:2211.04325. https://arxiv.org/abs/2211.04325
- Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*. arXiv:2210.03629. https://arxiv.org/abs/2210.03629
