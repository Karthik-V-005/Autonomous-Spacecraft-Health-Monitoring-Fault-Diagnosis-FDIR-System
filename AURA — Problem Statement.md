# AURA — Autonomous Spacecraft Health Monitoring, Fault Diagnosis & FDIR System

**Document:** Problem Statement  
**Version:** 1.0  
**Status:** Baseline / Frozen  
**Project Type:** Academic / Research Prototype  
**System:** AURA

---

## 1. Purpose

AURA is a software-based research prototype for autonomous spacecraft health management.

The system will investigate whether machine-learning-based telemetry analysis can assist with:

1. detection of anomalous spacecraft behavior;
2. identification of the subsystem or fault associated with an anomaly;
3. estimation of spacecraft/subsystem health;
4. selection of an appropriate predefined fault-response action; and
5. evaluation of the resulting spacecraft state after that action.

The project is intended to demonstrate an end-to-end spacecraft health-management workflow using publicly available spacecraft telemetry, a simulated spacecraft/digital-twin environment, machine-learning models, and a deterministic fault-response layer.

AURA is **not** intended to be flight-qualified software and must not be represented as such.

---

## 2. Real-World Motivation

Spacecraft operate with limited physical resources, limited opportunities for direct intervention, and potentially delayed communication with ground systems.

Space-system engineering therefore requires mechanisms for detecting abnormal system states, isolating faults, protecting critical functions, and recovering where the spacecraft design permits recovery.

NASA documentation explicitly identifies Fault Detection, Isolation, and Recovery (FDIR), health/status data, and autonomous operation as relevant spacecraft-system capabilities. NASA software-engineering guidance also describes autonomous fault-protection responses and spacecraft safing behavior. [1][2]

NASA/JPL has also demonstrated machine-learning approaches for spacecraft telemetry anomaly detection, including work involving Mars Science Laboratory telemetry. [3][4]

AURA uses these concepts as the engineering motivation for a student-scale research prototype.

---

## 3. Problem Definition

The central problem addressed by AURA is:

> Given spacecraft telemetry containing nominal and anomalous operating behavior, determine whether the spacecraft is operating outside its expected behavior, identify the most likely fault when sufficient information is available, and select a predefined recovery response within a simulated spacecraft environment.

The problem is divided into four technical problems:

### 3.1 Anomaly Detection

Determine whether an observed telemetry sequence is consistent with learned or explicitly defined nominal behavior.

Output:

```text
NORMAL
or
ANOMALY
```

with an anomaly score and associated timestamp/window.

---

### 3.2 Fault Diagnosis

When an anomaly is detected, determine the most likely fault hypothesis from the set of faults explicitly modeled by AURA.

Example:

```text
Observed anomaly
        ↓
Fault hypotheses
        ↓
Battery degradation       0.78
Thermal sensor fault      0.14
Thermal excursion         0.08
```

The probabilities/confidences produced by an ML model must be treated as model outputs, not as physically verified probabilities unless they have been appropriately calibrated.

---

### 3.3 Health Assessment

AURA will calculate a normalized health indicator for each modeled subsystem.

The health indicator is a **project-defined analytical metric**, not an actual spacecraft-certified health parameter.

Example:

```text
Power       0.94
Battery     0.81
Thermal     0.91
ADCS        0.97
Communications 0.95
```

The exact calculation will be defined in the requirements and implementation specifications.

---

### 3.4 Fault Response

After a fault has been detected and sufficiently isolated, AURA may select a predefined recovery action.

Examples include:

```text
Disable failed sensor
Switch to redundant sensor
Disable degraded actuator
Switch to redundant actuator
Reduce non-critical load
Enter safe configuration
Continue nominal operation
```

AURA will only execute recovery actions that have been explicitly modeled and validated in the simulated spacecraft.

The ML model will **not directly generate arbitrary spacecraft commands**.

---

## 4. Project Boundaries

### Included

AURA will include:

- spacecraft telemetry processing;
- time-series anomaly detection;
- classical anomaly-detection baselines;
- at least one deep-learning anomaly detector;
- a simulated spacecraft/digital twin;
- controlled fault injection;
- fault diagnosis;
- subsystem health estimation;
- deterministic FDIR logic;
- mission-control visualization;
- quantitative evaluation;
- reproducible experiments.

### Not Included

AURA will not attempt to provide:

- flight-qualified software;
- real spacecraft commanding;
- certification;
- real-time safety guarantees;
- complete spacecraft physics;
- complete spacecraft subsystem models;
- autonomous orbital operations;
- autonomous spacecraft navigation;
- autonomous mission planning;
- actual hardware-in-the-loop spacecraft control in the baseline version.

These may be considered future research extensions but are outside the frozen baseline scope.

---

## 5. Data Strategy

AURA will use two classes of data.

### 5.1 Real Spacecraft Telemetry

The primary real-data benchmark will be the publicly available NASA SMAP/MSL anomaly dataset associated with the work of Hundman et al.

The dataset contains telemetry streams from:

- NASA's Soil Moisture Active Passive (SMAP) satellite;
- NASA's Mars Science Laboratory (MSL) rover.

The published dataset contains 55 SMAP channels and 27 MSL channels, with labeled anomaly sequences. The data preserves individual telemetry-stream sequences but does not provide a complete synchronized spacecraft-wide telemetry record. [5][6]

Therefore, the dataset will primarily be used to evaluate:

> **AURA's telemetry anomaly-detection capability.**

It will not be treated as a complete spacecraft digital twin.

---

### 5.2 Simulated Spacecraft Telemetry

AURA will generate its own synchronized telemetry using a project-specific spacecraft digital twin.

This dataset will be used for:

- controlled fault injection;
- fault diagnosis;
- FDIR testing;
- recovery testing;
- end-to-end closed-loop demonstrations.

Every simulated parameter and fault model will be documented explicitly.

---

## 6. Engineering Philosophy

AURA will follow a hybrid architecture:

```text
Machine Learning
       +
Explicit engineering rules
       +
Simulated physical behavior
       +
Deterministic FDIR
```

Machine learning will primarily provide:

- anomaly scores;
- learned representations;
- fault hypotheses;
- predictive information.

The FDIR layer will use explicit, testable rules and state transitions.

This separation is intentional.

AURA will not allow an unconstrained ML model to directly issue arbitrary spacecraft commands.

---

## 7. Primary Research Question

> Can machine-learning-based analysis of spacecraft telemetry improve anomaly detection and fault diagnosis over simple threshold/statistical methods while remaining computationally practical and sufficiently interpretable for integration into a simulated spacecraft health-management system?

---

## 8. Secondary Research Questions

1. How do classical statistical methods compare with machine-learning-based anomaly detection?
2. Does temporal modeling improve detection of contextual anomalies?
3. How does anomaly-detection performance change under sensor noise and telemetry degradation?
4. Can multiple telemetry channels improve fault diagnosis in the simulated spacecraft?
5. How much detection latency is introduced by different ML models?
6. Can a deterministic FDIR layer safely use ML-generated fault hypotheses?
7. Which model provides the best trade-off between detection performance, false alarms, latency, and computational cost?

---

## 9. Success Definition

AURA will be considered successful if it demonstrates a reproducible end-to-end chain:

```text
Telemetry
    ↓
Preprocessing
    ↓
Anomaly Detection
    ↓
Fault Diagnosis
    ↓
Health Assessment
    ↓
FDIR Decision
    ↓
Simulated Recovery Action
    ↓
Post-Recovery Assessment
```

The system must be evaluated quantitatively rather than demonstrated only through screenshots.

---

## 10. Engineering Honesty Requirement

The project documentation must clearly distinguish:

### Real data

Data originating from publicly available NASA spacecraft/rover telemetry.

### Simulated data

Data generated by AURA's spacecraft digital twin.

### Learned behavior

Behavior inferred by ML models.

### Engineered behavior

Behavior explicitly encoded through spacecraft/fault/FDIR rules.

No simulated capability may be described as a capability demonstrated by the real NASA spacecraft dataset.

---

## References

[1] NASA, NPR 8705.2B, Chapter 3, spacecraft fault detection/isolation/recovery, health/status data, and autonomous operation.  
[2] NASA Software Engineering Handbook, Fault Detection and Response guidance.  
[3] NASA/JPL, "Telemetry Anomaly Detection System using Machine Learning to Streamline Mission Operations."  
[4] NASA/JPL, "MSL Telecom Automated Anomaly Detection."  
[5] Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding."  
[6] NASA SMAP/MSL telemetry anomaly dataset and associated public repository.

Primary NASA references and supporting sources are maintained in the project bibliography.