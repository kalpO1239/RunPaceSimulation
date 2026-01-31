# Running Pace Simulator (Dynamical Systems Model)
---

## Overview:

This project models running pace as an observation of a dynamical system, inspired by how weather models (e.g., the GFS) forecast future states rather than directly predicting observations. Similarly, instead of predicting speed outright, the runner is treated as a system with states—-fatigue and energy—-that evolve over time according to differential equations. Speed then emerges as an observation of these states.

---

## Core Idea:

### State variables:

Fatigue (F): accumulates with effort and decays over time

Energy (E): depletes with effort and replenishes at a baseline rate

Inputs: Stryd pod sensor data (power, form power ratio, ground contact time, cadence, grade)

### Dynamics:
The runner’s state updates each second via discretized ODEs (Euler Step).

### Observation model:
Speed is found from power and other biomechanics, modulated by the runner’s current fatigue and energy.

This mirrors weather modeling: the system state is propagated forward, and observations are derived from that state.

---

## What the Code Does:

Preprocesses raw Stryd CSV data and normalizes feature metrics

Simulates fatigue and energy evolution over time with discretized ODEs

Learns model parameters by minimizing prediction error (MSE) across multiple runs (training step)

Outputs predicted vs. observed speed

Allows for forecasting future speed and internal states starting from a given condition

---

## Interpretation & Limitations:

This model treats a runner as a forward-propagated dynamical system rather than a static regression problem. Speed is observed from state variables rather than direct prediction, allowing for internal states (fatigue and energy) to be observed over time.

Current limitations include limited training diversity (most runs I have on file are easy aerobic efforts) and simplified state dynamics. Ongoing work focuses on collecting higher-intensity data and refining the state update equations.

