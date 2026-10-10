# AURA — Autonomous Spacecraft Health Monitoring, Fault Diagnosis & FDIR System

**Document:** System Architecture  
**Version:** 1.0  
**Status:** Baseline / Frozen

---

# 1. Architectural Principle

AURA SHALL use a layered architecture in which:

- telemetry acquisition is separated from ML;
- anomaly detection is separated from diagnosis;
- diagnosis is separated from FDIR;
- FDIR is deterministic and constrained;
- the spacecraft digital twin is separated from the monitoring system;
- all decisions are logged.

The architecture SHALL prevent direct uncontrolled command generation by the ML system.

---

# 2. High-Level Architecture

```text
                         ┌─────────────────────────┐
                         │     DATA SOURCES         │
                         │                         │
                         │ NASA SMAP/MSL telemetry │
                         │ Simulated telemetry     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ TELEMETRY INGESTION     │
                         │                         │
                         │ loading                 │
                         │ validation              │
                         │ timestamp handling      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ TELEMETRY PROCESSING    │
                         │                         │
                         │ cleaning                │
                         │ missing data handling   │
                         │ normalization            │
                         │ window generation       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────────┐
                    │        HEALTH MONITORING            │
                    │                                    │
                    │ Classical detector                 │
                    │ Neural anomaly detector            │
                    │ Anomaly scoring                    │
                    └─────────────────┬──────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   FAULT DIAGNOSIS       │
                         │                         │
                         │ fault hypotheses       │
                         │ confidence/score        │
                         │ unknown state           │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   HEALTH ASSESSMENT     │
                         │                         │
                         │ subsystem health        │
                         │ system health           │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       FDIR ENGINE        │
                         │                         │
                         │ detection confirmation  │
                         │ isolation               │
                         │ response selection      │
                         │ recovery verification   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ SPACECRAFT DIGITAL TWIN │
                         │                         │
                         │ Power                   │
                         │ Battery                 │
                         │ Thermal                 │
                         │ ADCS                    │
                         │ Communications          │
                         └────────────┬────────────┘
                                      │
                                      │ simulated telemetry
                                      └───────────┐
                                                  │
                                                  ▼
                                      TELEMETRY PROCESSING
```

---

# 3. Major Components

AURA consists of the following logical components:

```text
AURA
├── Data Layer
├── Telemetry Layer
├── ML Layer
├── Diagnosis Layer
├── Health Layer
├── FDIR Layer
├── Simulation Layer
├── Visualization Layer
└── Experiment/Evaluation Layer
```

---

# 4. Data Layer

## Responsibility

The Data Layer provides access to:

1. real NASA telemetry;
2. simulated spacecraft telemetry;
3. anomaly/fault ground truth.

---

## Components

```text
data/
├── raw/
├── processed/
├── labels/
└── metadata/
```

### Raw

Original downloaded datasets.

### Processed

Derived datasets.

### Labels

Ground-truth anomaly/fault labels.

### Metadata

Dataset descriptions and preprocessing information.

---

# 5. Telemetry Ingestion Layer

## Responsibility

Convert source data into AURA's internal telemetry representation.

The internal representation SHALL contain at minimum:

```text
timestamp
parameter_id
value
validity
source
```

For simulated data, additional metadata MAY include:

```text
simulation_run_id
fault_id
subsystem
```

---

# 6. Telemetry Processing Layer

The processing pipeline is:

```text
Raw telemetry
      ↓
Validation
      ↓
Missing-value handling
      ↓
Outlier handling
      ↓
Normalization
      ↓
Temporal window generation
      ↓
ML-ready tensor
```

Processing SHALL be deterministic given the same configuration and input data.

---

# 7. Anomaly Detection Layer

The anomaly detector is divided into two classes.

## 7.1 Classical Detector

Examples:

- threshold detector;
- rolling-statistics detector;
- Isolation Forest.

This provides the baseline.

---

## 7.2 Neural Detector

Initial neural architecture:

```text
Telemetry Window
       ↓
Encoder / LSTM
       ↓
Latent representation
       ↓
Prediction or reconstruction
       ↓
Error
       ↓
Anomaly Score
```

The exact neural architecture is intentionally not frozen at this level.

The initial implementation SHALL use either:

- an autoencoder; or
- an LSTM-based detector.

A Transformer is an optional later experiment.

---

# 8. Anomaly Score Interface

Every detector SHALL expose a common conceptual interface:

```text
detect(window)
    →
{
    anomaly_score,
    anomaly_flag,
    timestamp/window,
    detector_id
}
```

This allows different models to be compared without modifying the rest of the system.

---

# 9. Fault Diagnosis Layer

The diagnosis layer receives:

```text
Telemetry
+
Anomaly information
+
Subsystem features
```

and produces:

```text
Fault hypothesis
+
confidence/score
+
affected subsystem
```

Conceptual interface:

```text
diagnose(observation)
    →
{
    fault_id,
    subsystem,
    confidence,
    evidence
}
```

The diagnosis layer SHALL support:

```text
UNKNOWN
```

when evidence is insufficient.

---

# 10. Health Assessment Layer

Health assessment converts telemetry and diagnosis information into project-defined health indicators.

Conceptual output:

```text
SubsystemHealth:
    power
    battery
    thermal
    adcs
    communications
```

Each value SHALL be normalized according to the health model specification.

Health scores SHALL not be interpreted as certified spacecraft health measurements.

---

# 11. FDIR Layer

The FDIR layer is deliberately separated from the ML layer.

```text
ML
 ↓
Evidence
 ↓
FDIR
 ↓
Action
```

not:

```text
ML
 ↓
arbitrary command
```

---

# 12. FDIR State Machine

Baseline state machine:

```text
                 ┌─────────────┐
                 │   NOMINAL   │
                 └──────┬──────┘
                        │
                 anomaly detected
                        │
                        ▼
                 ┌─────────────┐
                 │   SUSPECT   │
                 └──────┬──────┘
                        │
                confirmation /
                 diagnosis
                        │
                        ▼
              ┌──────────────────┐
              │ FAULT_ISOLATED   │
              └───────┬──────────┘
                      │
             ┌────────┴─────────┐
             │                  │
        recovery possible   recovery
             │              unavailable
             ▼                  │
       ┌──────────┐             ▼
       │ RECOVERY │          ┌──────┐
       └────┬─────┘          │ SAFE │
            │                └──────┘
       verification
            │
       ┌────┴─────┐
       │          │
    success    failure
       │          │
       ▼          ▼
    NOMINAL      SAFE
```

The transitions SHALL be deterministic.

---

# 13. Recovery Action Model

Every recovery action shall have the following structure:

```text
Action:
    action_id
    triggering_fault
    affected_subsystem
    preconditions
    command
    expected_response
    verification_condition
    timeout
    failure_action
```

Example:

```text
action_id:
    ADCS_SWITCH_BACKUP_WHEEL

triggering_fault:
    RW02_DEGRADED

precondition:
    RW03_AVAILABLE

command:
    DISABLE_RW02
    ENABLE_RW03

verification:
    attitude_error < configured_limit
    for configured_duration
```

The numerical limits SHALL be defined in the digital-twin configuration rather than embedded as unexplained constants in the code.

---

# 14. Spacecraft Digital Twin

The digital twin is a simplified state-based spacecraft model.

```text
                   SPACECRAFT STATE
                         │
       ┌─────────────────┼──────────────────┐
       │                 │                  │
       ▼                 ▼                  ▼
     POWER             THERMAL            ADCS
       │                 │                  │
       ▼                 ▼                  ▼
    BATTERY        COMPONENT TEMP      ATTITUDE
       │
       └─────────────────┬──────────────────┘
                         ▼
                    TELEMETRY
```

---

# 15. Digital-Twin State

The baseline digital twin SHALL maintain state for:

## Power

```text
generated_power
load_power
net_power
```

## Battery

```text
state_of_charge
voltage
current
temperature
```

## Thermal

```text
component_temperature
heat_input
```

## ADCS

```text
attitude_error
angular_rate
reaction_wheel_state
```

## Communications

```text
data_queue
packet_loss_rate
link_available
```

The exact state vector SHALL be frozen in the implementation specification before the simulator is finalized.

---

# 16. Simulation Loop

The digital twin SHALL execute conceptually as:

```text
state(t)
   ↓
calculate subsystem behavior
   ↓
apply disturbances/faults
   ↓
calculate state(t+1)
   ↓
generate telemetry
   ↓
send telemetry to AURA
```

For closed-loop experiments:

```text
telemetry
   ↓
AURA
   ↓
FDIR action
   ↓
digital twin
   ↓
new state
   ↓
new telemetry
```

---

# 17. Fault Injection Architecture

Fault injection SHALL occur inside the simulation layer.

```text
                    Digital Twin
                         │
                         ▼
                   Fault Manager
                         │
       ┌─────────────────┼────────────────┐
       ▼                 ▼                ▼
   Sensor fault      Battery fault     ADCS fault
       │                 │                │
       └─────────────────┼────────────────┘
                         ▼
                    State Model
                         │
                         ▼
                      Telemetry
```

A fault injection SHALL not directly modify ML predictions.

It shall modify the simulated system state or sensor behavior.

---

# 18. Fault Lifecycle

Every simulated fault follows:

```text
UNINJECTED
    ↓
INJECTED
    ↓
DEVELOPING
    ↓
OBSERVABLE
    ↓
DETECTED
    ↓
DIAGNOSED
    ↓
RECOVERY ATTEMPTED
    ↓
RECOVERY VERIFIED
```

Not every fault must pass through every state.

For example:

```text
FAULT
 ↓
DETECTED
 ↓
NO RECOVERY AVAILABLE
 ↓
SAFE
```

---

# 19. Mission-Control Layer

The visualization layer shall consume system events and telemetry.

It SHALL display:

```text
Spacecraft state
Subsystem health
Telemetry
Anomaly score
Fault diagnosis
FDIR state
Recovery action
Event history
```

The dashboard is a visualization and operator-assistance component.

It does not replace the underlying monitoring/FDIR logic.

---

# 20. Optional Open MCT Integration

Open MCT may be used as the mission-control visualization framework.

NASA describes Open MCT as an open-source mission-control/telemetry visualization framework with support for telemetry sources and extensible plugins. [7]

If Open MCT is used, AURA-specific ML and FDIR logic SHALL remain external modules.

Conceptually:

```text
AURA
 │
 ├── ML
 ├── Diagnosis
 ├── FDIR
 │
 └── Telemetry API
          │
          ▼
      Open MCT
```

Open MCT is therefore a visualization/integration layer, not the AURA decision engine.

---

# 21. Experiment Layer

The experiment system SHALL support:

```text
experiment configuration
        ↓
dataset/simulation selection
        ↓
model configuration
        ↓
execution
        ↓
metrics
        ↓
results
```

Every experiment SHALL record:

```text
experiment_id
dataset
model
model_version
configuration
random_seed
metrics
timestamp
```

---

# 22. Evaluation Architecture

The evaluation system shall compare:

```text
                  ANOMALY DETECTION
                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
    Threshold      Classical ML        Deep ML
                         │
                         ▼
                  Common Metrics
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    Precision          Recall             F1
```

For the simulated spacecraft:

```text
Injected Fault
      ↓
Ground Truth
      ↓
AURA Detection
      ↓
AURA Diagnosis
      ↓
AURA Recovery
      ↓
Compare with Ground Truth
```

---

# 23. Recommended Software Boundaries

The implementation SHOULD follow these boundaries:

```text
src/
├── data/
├── telemetry/
├── models/
├── detection/
├── diagnosis/
├── health/
├── fdir/
├── simulation/
├── evaluation/
└── interface/
```

Responsibilities:

### `data`

Dataset loading and metadata.

### `telemetry`

Internal telemetry representation and preprocessing.

### `models`

ML models.

### `detection`

Anomaly scoring.

### `diagnosis`

Fault classification/hypothesis generation.

### `health`

Health estimation.

### `fdir`

Deterministic fault-response logic.

### `simulation`

Digital twin and fault injection.

### `evaluation`

Metrics and experiments.

### `interface`

Dashboard/API.

---

# 24. Data Flow Contract

The major interfaces should follow these conceptual contracts.

### Telemetry → Detector

```text
timestamp
parameter/window
values
metadata
```

### Detector → Diagnosis

```text
anomaly_flag
anomaly_score
affected_window
features/evidence
```

### Diagnosis → FDIR

```text
fault_id
subsystem
confidence
evidence
```

### FDIR → Simulator

```text
action_id
parameters
timestamp
```

### Simulator → Telemetry

```text
new spacecraft state
↓
telemetry
```

---

# 25. Architectural Safety Boundary

The following boundary is mandatory:

```text
                 ┌─────────────────┐
                 │   ML SYSTEM     │
                 │                 │
                 │ anomaly score   │
                 │ fault hypothesis│
                 └────────┬────────┘
                          │
                     evidence
                          │
                          ▼
                 ┌─────────────────┐
                 │   FDIR ENGINE   │
                 │                 │
                 │ deterministic   │
                 │ rules/states    │
                 └────────┬────────┘
                          │
                    validated
                     action ID
                          │
                          ▼
                 ┌─────────────────┐
                 │ DIGITAL TWIN    │
                 └─────────────────┘
```

The ML layer SHALL NOT bypass the FDIR layer.

---

# 26. Initial Technology Stack

The baseline implementation SHOULD use:

### Programming

- Python 3.x

### Scientific computing

- NumPy
- SciPy
- Pandas

### ML

- scikit-learn
- PyTorch

### Visualization

- Matplotlib
- Plotly

### Dashboard

- Streamlit initially

### Version control

- Git
- GitHub

### Testing

- pytest

Additional technologies SHALL only be introduced when they solve a demonstrated project requirement.

---

# 27. Architecture Evolution

The architecture SHALL evolve in this order:

```text
Version 0
Telemetry exploration
        ↓
Version 1
Anomaly detection
        ↓
Version 2
Digital twin
        ↓
Version 3
Fault injection
        ↓
Version 4
Fault diagnosis
        ↓
Version 5
FDIR
        ↓
Version 6
Closed-loop simulation
        ↓
Version 7
Mission-control interface
        ↓
Version 8
Advanced ML / research experiments
```

Advanced technologies SHALL NOT be allowed to destabilize the baseline architecture.

---

# 28. Architectural Non-Goals

The baseline architecture does not attempt to model:

- complete spacecraft orbital dynamics;
- detailed propulsion;
- detailed thermal finite-element behavior;
- real flight software;
- real spacecraft command protocols;
- certified safety systems;
- radiation-hardened computing;
- real spacecraft communications;
- complete spacecraft hardware redundancy.

Such capabilities require substantially more mission-specific information and are outside the scope of the baseline student prototype.

---

# 29. Design Principle

The central architectural principle of AURA is:

> **Use real spacecraft data where it provides real evidence, use simulation where controlled experimentation is required, use ML where learned behavior is useful, and use deterministic engineering logic where safety and recovery decisions must be constrained.**

This separation is a fundamental requirement of the project.