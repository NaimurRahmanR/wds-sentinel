# Architecture

## Executed pipeline

```text
BattLeDIM pressure / flow / level
        |
        v
Evidence adapters -> Evidence Agent -------------------------+
        |                                                     |
Frozen predictor -> Prediction Agent                         |
        |                                                     v
        +----------------> Knowledge / Reasoning Agent -> Reliability Agent
                                                      |              |
                                                      +--------------+
                                                                     v
                                                        Supervisory Decision
                                                                     |
                                                                     v
                                                           DecisionRecord + XAI

Independent CWA/CODIS precipitation -> Hazard Agent -> Hazard KBS ----+
                                                                      |
                                                                      v
                                                    IntegratedSupervisoryAgent
                                                                      |
                                                                      v
                                               IntegratedDecisionRecord / action
```

The precipitation stream is independent of BattLeDIM/L-Town. It contributes monitoring context but never modifies the frozen WDS operational decision or creates a rainfall-to-hydraulic causal claim.

## Components

- `prediction/` — causal feature engineering, frozen Logistic Regression predictor, deterministic degradation utilities and event metrics.
- `evidence/` — typed `EvidenceState` / `EvidenceBundle` objects plus adapter protocol for raw sensor, prediction and hazard evidence.
- `knowledge/` — deterministic `Fact`, `Rule`, `KnowledgeBase` and `ReasoningTrace` with identifiable WDS and hazard rules.
- `reasoning/` — frozen direct/hybrid corroboration logic used by the explicit KBS.
- `reliability/` — missingness, disagreement and instability signals plus the fixed selective-control policy.
- `agents/` — Evidence, Prediction, Knowledge/Reasoning, Reliability, Hazard, WDS Supervisory and Integrated Supervisory agents with typed message contracts.
- `hazard/` — precipitation loading, trace-value handling and training-calibrated multi-window extreme indicator.
- `experiments/` — split/experiment utilities and helpers that construct agent pipelines from actual experiment outputs.
- `explanation/` / `DecisionRecord` — grounded templated explanations based only on executed rules, evidence availability and reliability reasons.

## Non-bypass guarantees

The final WDS decision is always the `ReliabilityMessage.final_decision`; the Supervisory Agent never emits the reasoning-layer decision directly. Unit tests exercise this property with a stub reliability agent.

The Integrated Supervisory Agent calls the full WDS `execute()` path and the Hazard Agent itself. Separate WDS and hazard timestamps are supported because the bundled streams are deliberately unrelated; this avoids the former manual assembly path used by early demonstration code.

## Real integrated scenarios

The current scenario exporter selects actual 2018 BattLeDIM validation rows:

- a true normal/no-onset row with `NO_ALERT`;
- a true onset-window row that the frozen clean system actually detects as `ALERT`;
- a true onset-window row under the fixed missing-sensor degradation that invokes `ABSTAIN`/`ESCALATE`.

Those rows are paired, as independent contexts, with actual 2018 precipitation days classified as `NORMAL` or `ELEVATED`. No WDS score, corroboration count, missingness value or sensor reading is hand-authored in the final integrated experiment.
