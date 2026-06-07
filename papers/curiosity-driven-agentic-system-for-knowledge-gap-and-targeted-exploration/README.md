# curiosity-driven agentic system

**[A curiosity-driven agentic system for knowledge gap identification and targeted exploration](A_Curiosity_Driven_Agentic_System_for_Knowledge_Gap_Identification_and_Targeted_Exploration.pdf)** · whitepaper · February 2026

LLMs answer fluently whether or not they actually know. This whitepaper proposes a system that treats "not knowing" as a first-class state: a structured knowledge model with an explicit known/unknown/uncertain partition, an LLM reasoning core for task parsing and gap detection, a planning module that converts gaps into targeted actions — retrieval, simulation, experimentation, inspection — and a feedback loop that writes validated findings back into the knowledge model. Includes subsystem interfaces, data flow, operational constraints, evaluation protocols, and case studies across space exploration, scientific research, industrial automation, healthcare, and education.

This is the umbrella architecture for the rest of the research in this repository: the simulated-world-models papers specify the world-model component, and the sig-reg-dynamics study is empirical work on training such a component.

PDF only — no markdown source.
