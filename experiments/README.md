# Experiments

This directory contains research harnesses, prototypes, and implementation-risk tests.

## Current organization

```text
v0/mechanistic_simulator/   Legacy V0 volume-only simulator and recovery harness
prior_builder/             Planned V1 layered-prior implementation experiments
twin_runtime/             V1 posterior-update and scenario-runtime prototypes
```

The V0 harness remains useful as a regression baseline, but it should not keep growing into the V1 architecture. V1 work should keep prior construction, simulator adapters, benchmark runners, and generated evaluation reports separate enough that each layer can be ablated and compared against simple baselines.

The repository may temporarily keep a compatibility symlink at `experiments/mechanistic_simulator` pointing to `experiments/v0/mechanistic_simulator` so existing scripts and imports continue to work during the transition.

See `../roadmap/V1_PRIOR_STACK_IMPLEMENTATION_AND_EVALUATION_PLAN.md` for the recommended V1 build and evaluation plan.