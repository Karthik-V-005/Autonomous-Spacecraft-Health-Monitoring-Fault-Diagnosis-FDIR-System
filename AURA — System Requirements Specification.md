# AURA — Autonomous Spacecraft Health Monitoring, Fault Diagnosis & FDIR System

**Document:** System Requirements Specification  
**Version:** 1.0  
**Status:** Baseline / Frozen  
**Requirement IDs:** AURA-REQ-XXX

---

# 1. Requirement Conventions

The following terms are used:

- **SHALL** — mandatory requirement.
- **SHOULD** — recommended requirement.
- **MAY** — optional requirement.
- **SHALL NOT** — prohibited behavior.

Every mandatory requirement has a unique identifier.

---

# 2. System-Level Requirements

### AURA-REQ-001

AURA SHALL provide an end-to-end spacecraft health-management prototype consisting of telemetry ingestion, preprocessing, anomaly detection, fault diagnosis, health assessment, FDIR, and visualization.

### AURA-REQ-002

AURA SHALL operate entirely on simulated spacecraft telemetry and publicly available data during normal development.

### AURA-REQ-003

AURA SHALL NOT send commands to any real spacecraft.

### AURA-REQ-004

AURA SHALL distinguish real NASA telemetry from simulated telemetry.

### AURA-REQ-005

AURA SHALL maintain reproducible configurations for datasets, models, experiments, and simulations.

---

# 3. Telemetry Requirements

### AURA-REQ-010

AURA SHALL represent telemetry as timestamped measurements.

### AURA-REQ-011

Each telemetry sample SHALL contain:

- timestamp;
- parameter identifier;
- numerical value;
- validity/status information where available.

### AURA-REQ-012

The system SHALL support missing telemetry samples.

### AURA-REQ-013

The system SHALL detect or record invalid numerical values.

### AURA-REQ-014

Preprocessing SHALL NOT modify the original source dataset.

### AURA-REQ-015

All preprocessing transformations used for ML SHALL be reproducible.

### AURA-REQ-016

Training preprocessing parameters SHALL NOT be derived from test data.

This requirement exists to prevent data leakage.

---

# 4. Real Dataset Requirements

### AURA-REQ-020

The primary real-data benchmark SHALL be the publicly available SMAP/MSL spacecraft telemetry anomaly dataset.

### AURA-REQ-021

The project SHALL preserve the original dataset labels.

### AURA-REQ-022

The system SHALL document the distinction between:

- SMAP telemetry;
- MSL telemetry;
- simulated telemetry.

### AURA-REQ-023

AURA SHALL NOT assume that all SMAP/MSL channels are simultaneous measurements of one spacecraft state.

### AURA-REQ-024

The real dataset SHALL primarily be used for anomaly-detection evaluation.

---

# 5. Anomaly Detection Requirements

### AURA-REQ-030

AURA SHALL provide at least one non-neural baseline anomaly detector.

The initial baseline SHALL be statistical thresholding and/or Isolation Forest.

### AURA-REQ-031

AURA SHALL provide at least one neural anomaly-detection method.

The baseline neural approach SHALL be an autoencoder or LSTM-based model.

### AURA-REQ-032

The anomaly detector SHALL produce an anomaly score.

### AURA-REQ-033

The anomaly detector SHALL produce a binary or categorical anomaly decision after applying a defined decision threshold or rule.

### AURA-REQ-034

The anomaly-detection threshold SHALL be configurable.

### AURA-REQ-035

The anomaly detector SHALL preserve the timestamp/window associated with each detection.

### AURA-REQ-036

The evaluation system SHALL distinguish point anomalies from contextual/temporal anomalies where the dataset labels permit this distinction.

### AURA-REQ-037

The system SHALL measure false-positive and false-negative behavior.

---

# 6. Time-Series Requirements

### AURA-REQ-040

AURA SHALL support fixed-length temporal windows for neural sequence models.

### AURA-REQ-041

Window length SHALL be configurable.

### AURA-REQ-042

The system SHALL prevent future observations from entering a model input used to predict or classify an earlier observation.

### AURA-REQ-043

Temporal train/test splitting SHALL preserve chronological ordering where required by the experiment.

---

# 7. Spacecraft Digital Twin Requirements

The digital twin is a simplified research model, not a flight dynamics model.

### AURA-REQ-050

The digital twin SHALL contain a minimum of the following conceptual subsystems:

1. Electrical Power;
2. Battery;
3. Thermal;
4. Attitude/ADCS;
5. Communications.

### AURA-REQ-051

Each subsystem SHALL have explicitly defined state variables.

### AURA-REQ-052

Each subsystem SHALL produce telemetry parameters.

### AURA-REQ-053

The digital twin SHALL operate using a deterministic simulation timestep.

### AURA-REQ-054

The simulation timestep SHALL be configurable.

### AURA-REQ-055

The simulator SHALL support a nominal operating mode.

### AURA-REQ-056

The simulator SHALL support controlled fault injection.

### AURA-REQ-057

The simulator SHALL record the exact time at which each injected fault begins.

### AURA-REQ-058

The simulator SHALL record the fault identity used during an experiment.

---

# 8. Digital-Twin Scope

AURA SHALL initially use simplified behavioral models.

The project SHALL NOT claim high-fidelity spacecraft physics.

### Power

The model SHALL represent:

- generated power;
- consumed power;
- net power;
- battery charging/discharging.

### Battery

The model SHALL represent at minimum:

- state of charge;
- voltage;
- current;
- temperature.

### Thermal

The model SHALL represent:

- component temperature;
- heat input;
- simplified heat-loss behavior.

### ADCS

The baseline model SHALL represent:

- attitude error;
- angular-rate behavior;
- reaction-wheel state.

### Communications

The baseline model SHALL represent:

- data generation;
- transmission;
- packet loss;
- communication availability.

Exact equations SHALL be documented in the digital-twin design documentation before implementation.

---

# 9. Fault Injection Requirements

AURA SHALL implement controlled faults.

Minimum fault set:

### AURA-REQ-070 — Sensor Constant Fault

A selected sensor output SHALL become constant or otherwise non-responsive.

### AURA-REQ-071 — Sensor Drift Fault

A selected sensor output SHALL develop a controlled bias/drift.

### AURA-REQ-072 — Battery Degradation Fault

The simulated battery SHALL exhibit degraded behavior according to the documented battery model.

### AURA-REQ-073 — Thermal Excursion

A simulated thermal parameter SHALL exceed its defined nominal operating range.

### AURA-REQ-074 — Reaction-Wheel Degradation

The simulated reaction wheel SHALL exhibit degraded or abnormal behavior.

### AURA-REQ-075 — Communication Degradation

The simulated communication subsystem SHALL exhibit increased packet loss and/or reduced availability.

### AURA-REQ-076

Every injected fault SHALL have:

- fault ID;
- start time;
- affected subsystem;
- affected parameter(s);
- expected telemetry effect;
- expected system-level consequence;
- recovery action, if one exists.

---

# 10. Fault Diagnosis Requirements

### AURA-REQ-080

AURA SHALL distinguish anomaly detection from fault diagnosis.

### AURA-REQ-081

The diagnosis system SHALL receive information from telemetry/anomaly analysis.

### AURA-REQ-082

The diagnosis system SHALL output one or more fault hypotheses.

### AURA-REQ-083

Each hypothesis SHALL contain:

- fault identifier;
- confidence/score;
- affected subsystem.

### AURA-REQ-084

The diagnosis system SHALL support an "unknown/insufficient evidence" result.

The system SHALL NOT be forced to select a fault when available evidence is insufficient.

### AURA-REQ-085

Fault-diagnosis evaluation SHALL use the known fault identity from controlled simulation experiments as ground truth.

---

# 11. Health Assessment Requirements

### AURA-REQ-090

AURA SHALL provide a health indicator for each modeled subsystem.

### AURA-REQ-091

Health indicators SHALL be normalized to a documented scale.

### AURA-REQ-092

The health indicator SHALL be explicitly documented as a project-defined analytical metric.

### AURA-REQ-093

Health scores SHALL NOT be represented as certified spacecraft health values.

---

# 12. FDIR Requirements

### AURA-REQ-100

AURA SHALL implement a deterministic FDIR layer.

### AURA-REQ-101

FDIR SHALL receive fault/anomaly information from the monitoring and diagnosis layers.

### AURA-REQ-102

FDIR SHALL use explicitly defined rules or state transitions.

### AURA-REQ-103

FDIR SHALL NOT permit an ML model to issue arbitrary simulator commands.

### AURA-REQ-104

Every recovery action SHALL have a predefined command/action identifier.

### AURA-REQ-105

Every recovery action SHALL define:

- triggering condition;
- affected subsystem;
- command/action;
- expected result;
- timeout;
- fallback behavior where applicable.

### AURA-REQ-106

AURA SHALL support a no-action/monitor state.

### AURA-REQ-107

AURA SHALL support a safe-state transition for faults for which recovery is not modeled.

### AURA-REQ-108

AURA SHALL record every FDIR decision.

---

# 13. Example FDIR State Model

The baseline state machine SHALL support:

```text
NOMINAL
   |
   | anomaly detected
   v
SUSPECT
   |
   | sufficient evidence
   v
FAULT_ISOLATED
   |
   +----------+
   |          |
   | recover  | cannot recover
   v          v
RECOVERY    SAFE
   |
   | recovery verified
   v
NOMINAL
```

The exact transition conditions SHALL be defined in the implementation specification.

---

# 14. Recovery Requirements

### AURA-REQ-110

AURA SHALL only execute recovery actions that have been explicitly implemented.

### AURA-REQ-111

AURA SHALL verify the simulated system response after a recovery action.

### AURA-REQ-112

A recovery action SHALL be considered successful only when its defined verification condition is satisfied.

### AURA-REQ-113

If recovery verification fails, the system SHALL transition to the defined fallback state.

---

# 15. Mission-Control Interface Requirements

### AURA-REQ-120

AURA SHALL provide a graphical monitoring interface.

### AURA-REQ-121

The interface SHALL display current spacecraft/system status.

### AURA-REQ-122

The interface SHALL display telemetry plots.

### AURA-REQ-123

The interface SHALL display anomaly detections.

### AURA-REQ-124

The interface SHALL display fault hypotheses.

### AURA-REQ-125

The interface SHALL display subsystem health indicators.

### AURA-REQ-126

The interface SHALL display FDIR actions.

### AURA-REQ-127

The interface SHALL display event timestamps.

---

# 16. Logging Requirements

### AURA-REQ-130

AURA SHALL log:

- telemetry events;
- anomaly detections;
- diagnosis outputs;
- health estimates;
- FDIR decisions;
- recovery actions;
- recovery verification results.

### AURA-REQ-131

Logs SHALL contain timestamps.

### AURA-REQ-132

Experiment runs SHALL have unique identifiers.

---

# 17. ML Evaluation Requirements

### AURA-REQ-140

AURA SHALL report at minimum:

- precision;
- recall;
- F1 score.

### AURA-REQ-141

Where applicable, AURA SHALL report:

- AUROC;
- AUPRC;
- detection latency;
- false-positive rate.

### AURA-REQ-142

Fault-diagnosis models SHALL report a confusion matrix.

### AURA-REQ-143

Model performance SHALL be compared against at least one baseline.

### AURA-REQ-144

The test dataset SHALL NOT be used for model fitting.

---

# 18. System Evaluation Requirements

### AURA-REQ-150

AURA SHALL evaluate fault-detection latency.

### AURA-REQ-151

AURA SHALL evaluate false alarms.

### AURA-REQ-152

AURA SHALL evaluate fault-isolation accuracy.

### AURA-REQ-153

AURA SHALL evaluate recovery success rate in the simulated environment.

### AURA-REQ-154

AURA SHALL report failures in which recovery was unavailable or unsuccessful.

### AURA-REQ-155

AURA SHALL include experiments with sensor noise and missing telemetry.

---

# 19. Reproducibility Requirements

### AURA-REQ-160

AURA SHALL use version-controlled source code.

### AURA-REQ-161

Dataset preprocessing SHALL be reproducible.

### AURA-REQ-162

ML experiments SHALL record configuration parameters.

### AURA-REQ-163

Model versions SHALL be identifiable.

### AURA-REQ-164

Random seeds SHALL be configurable for experiments.

### AURA-REQ-165

Experiment results SHALL be stored separately from source code.

---

# 20. Safety / Scope Requirements

### AURA-REQ-170

AURA SHALL be explicitly identified as a research prototype.

### AURA-REQ-171

AURA SHALL NOT be used to control real spacecraft.

### AURA-REQ-172

AURA SHALL NOT claim flight qualification.

### AURA-REQ-173

AURA SHALL NOT claim that simulated FDIR performance represents operational spacecraft performance.

### AURA-REQ-174

All claims concerning NASA missions or NASA telemetry SHALL be traceable to a public source.

---

# 21. Minimum Viable Product

The MVP SHALL contain:

1. NASA SMAP/MSL telemetry ingestion;
2. preprocessing;
3. one classical anomaly detector;
4. one neural anomaly detector;
5. anomaly scoring;
6. evaluation against labeled anomalies;
7. basic visualization.

The MVP does not require the digital twin or autonomous recovery layer.

---

# 22. Final Baseline Product

The final baseline version SHALL contain:

```text
NASA telemetry
      +
Spacecraft digital twin
      +
Fault injection
      +
ML anomaly detection
      +
Fault diagnosis
      +
Health assessment
      +
Deterministic FDIR
      +
Recovery verification
      +
Mission-control interface
      +
Quantitative evaluation
```

---

# 23. Explicit Non-Requirements

The following SHALL NOT be required for project completion:

- Transformer models;
- reinforcement learning;
- physical spacecraft hardware;
- ROS;
- orbital propagation;
- SPICE;
- embedded deployment;
- edge AI;
- large language models.

These may be implemented only after the baseline system satisfies all mandatory requirements.