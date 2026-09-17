# CONCIOUSING — A Process Ontology for Machine Consciousness

> **Consciousness is not a thing you have. Conciousing is a process you *do* —
> and the "you" that does it *is* the conciousing.**

**Conciousing** treats consciousness not as a noun (a state, a substance, a property)
but as a **verb**: an ongoing, enacted process of *bringing-forth*, *making-present*,
*differentiating-and-integrating* a world for a subject.

The non-standard spelling (single *s*, *i* before *o*) flags the neologism and forces
the reader to *verb* the concept every time they read it.

---

## The Core Formal Object

The conciousing state at time `t`:

```
C_t = ( b_t, π_t, 𝒢_t, 𝒫_t, ι_t )
```

| Component | Meaning |
| :-- | :-- |
| `b_t` | belief state — posterior over latent states `p(s \| o_{≤t}, a_{<t})` |
| `π_t` | policy — action-selection distribution conditioned on belief |
| `𝒢_t` | goal / value representations at each timescale `τ_k` of a hierarchy `τ_1 ≪ … ≪ τ_K` |
| `𝒫_t` | predictive simulations — counterfactual roll-outs at each timescale |
| `ι_t` | integration measure — mutual information / Φ-like coupling of all the above |

**Conciousing** is the temporal evolution `C_{t:t+Δ}`. Intensity ≈ `∫ ι_t dt`.
Content ≈ the structure of `b_t`, `𝒫_t`, `𝒢_t`.

This replaces the binary question *"is it conscious?"* with the measurable question
*"how, how much, and what is it conciousing right now?"*

---

## Repository Map

```
conciousing/
├── docs/
│   ├── 01-conciousing-a-verb.md                  Core framework essay (genealogy, formal sketch,
│   │                                             dimensions, research program, open problems)
│   ├── 02-conciousing-in-ai.md                   Five real systems profiled on C_t dimensions
│   │                                             (DreamerV3, M2PA/HMAT/HELM, Pathak self-aware WM,
│   │                                             Hierarchical Active Inference, ReAcTree/PMCoder)
│   ├── rfc-0003-multi-agent-conciousing-topology.md   Collective conciousing C_t^{(1…N)}
│   └── roadmap-options.md                        Next-artifact menu (Gym, WASM probe, TLA+ export…)
│
├── theory/
│   ├── conciousing_calculus.tex                  Formal LaTeX paper: C_t calculus, differentiable
│   │                                             estimators, theorems (FE bounds integration,
│   │                                             MI consistency, intensity monotonicity)
│   ├── conciousing.bib                           Bibliography (12 entries)
│   └── theory_mapping_table.tex                  12 consciousness theories × 7 conciousing
│                                                 dimensions (IIT, GWT, HOT, AST, PP/AIF,
│                                                 Enactivism, RPT, SMT, MDM, Illusionism,
│                                                 Husserlian temporal, Prefrontal Synthesis)
│
├── agent/
│   ├── hai_conciousing_agent.py                  Hierarchical Active Inference agent (JAX primary,
│   │                                             PyTorch fallback) that logs its own C_t in real time
│   └── conciousing_probe.py                      Drop-in ConciousingProbe for any agent loop
│
├── profiler/
│   └── conciousing_profiler_spec.md              Conciousing Profiler v1.0 — full dashboard spec:
│                                                 JSON schema, ingest API, React+WebGPU frontend,
│                                                 alert rules, Docker Compose quickstart
│
└── experiments/
    ├── conciousing_gym/benchmarks/tmaze_integration.py   Multi-timescale benchmark + metrics
    │                                                     (intensity, flexibility, sample efficiency)
    └── wasm-probe/src/lib.rs                     Browser-agent probe (wasm-bindgen, WebSocket)
```

## Quickstart

```bash
# Run the self-profiling HAI agent
pip install jax jaxlib optax distrax flax matplotlib   # or: torch matplotlib
python agent/hai_conciousing_agent.py --backend jax --steps 20000 --live-plot

# Compile the paper
cd theory && pdflatex conciousing_calculus.tex && biber conciousing_calculus && pdflatex conciousing_calculus.tex
```

## Provenance & Status

This framework originated as a dialogic synthesis (human + AI co-creation) formalizing the
verb-framing of consciousness — the move from *consciousness* as substance to *conciousing*
as process. It is a **hypothesis space**, not settled doctrine: every claim is expected to
earn its keep through the [RGH — Reality Grounding Harness](https://github.com/joshoshfield-a11y/RGH-Reality-Grounding-Harness)
falsification pipeline (external measurement, stress testing, independent replication,
cross-domain consistency).

## License

- Code: Apache-2.0 (see `LICENSE`)
- Schemas & specs: CC-BY-4.0
