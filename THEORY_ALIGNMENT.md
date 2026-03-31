**Theory Alignment**

This file turns the Nexus Trader theory into the exact implementation target for the repo.

**Foundation**

The intended foundation is:

1. `World Layer`
2. `Perception Layer`
3. `Simulation Layer`
4. `Brain Layer`
5. `Future Branching`
6. `Reverse Collapse`
7. `Probability Cone UI`

That is the target architecture. The repo should stay aligned to this even when implementation happens in smaller steps.

**World Layer**

Target inputs:

- 5-minute OHLCV for the main traded symbols
- breaking macro and gold-related headlines
- macro state time series
- crowd and narrative data
- optional daily LLM macro thesis used only as an interpretable bias input

Implementation rule:

- The LLM is a text reader, not the predictor.
- News, crowd, and macro stay independent until later fusion.

**Perception Layer**

Target outputs per timestep:

- price features
- news embedding
- crowd embedding
- macro context features

Implementation rule:

- FinBERT or sentence-transformer style encoders can be used as frozen translators.
- Reduction to compact latent vectors is allowed, but timestamp alignment must remain inspectable.

**Simulation Layer**

Target personas:

- retail
- institutional
- algo or HFT
- whale
- noise or civilian

Implementation rule:

- personas do not vote equally
- personas may use different strategy primitives
- ICT, SMC, Wyckoff, support or resistance, momentum, and macro bias belong here as persona behavior, not as one giant opaque model

**Brain Layer**

Target state:

- the forecasting model consumes fused sequential features and simulation context
- multi-horizon forecasting is the end-state goal
- the initial repo can use a simpler binary or short-horizon model, but the direction is toward a richer multi-horizon TFT-style forecaster

**Future Branching**

Target state:

- forward expansion from the current market state into multiple plausible futures
- branch weights come from historical consistency, current regime fit, and persona coherence
- pruning is allowed, but all surviving leaves should still contribute to the final collapse

**Reverse Collapse**

Target state:

- all surviving leaves act like weighted voters
- output includes weighted mean path, uncertainty width, and directional confidence
- disagreement is preserved, not hidden

**UI**

Target state:

- historical candles on the left
- future cone on the right
- consensus path, cone width, persona contribution, dominant drivers
- minority or risk scenarios can be rendered later as ghost cones

**What This Means For Build Order**

1. Get the environment and directory tree stable.
2. Download raw datasets fast and reproducibly.
3. Build price, macro, news, and crowd perception artifacts.
4. Build persona simulation and branch expansion.
5. Train the sequential forecasting model.
6. Collapse branches into cone outputs.
7. Render the UI and validate the full path.

**Accuracy Interpretation**

The goal is not a magical headline model accuracy number. The real goal is that filtered predictions are correct often enough to be useful after costs, with uncertainty exposed honestly. If observed accuracy becomes extremely high, leakage and target contamination must be ruled out before trusting it.
