import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------------
# PARAMETERS (initial guesses)
# -------------------------
alpha = 0.05
delta = 0.02
epsilon = 0.01
beta = 0.05

gamma = 0.001
phi = 0.01
eta = 0.1

# Constants to train
kappa0 = 3.5
kappa2 = 0.1
kappa3 = 0.1
kappa4 = 0.005
lam = 1.2

dt = 1  # seconds

speed_bias = 0.5
# -------------------------
# STATE VECTOR
# -------------------------
class RunnerState:
    def __init__(self, fatigue=0.0, energy=1.0):
        self.F = fatigue
        self.E = energy

# -------------------------
# DISCRETIZED ODE UPDATE
# -------------------------
def update_state(state, Pt, FPRt, GCTt, grade_t):
    F_next = state.F + dt * (
        alpha * Pt
        + delta * FPRt
        + epsilon * GCTt
        - beta * state.F
    )

    E_next = state.E + dt * (
        -gamma * Pt * (1 + eta * abs(grade_t))
        + phi
    )

    E_next = np.clip(E_next, 0.0, 1.0)
    return RunnerState(F_next, E_next)

# -------------------------
# NORMALIZE FEATURES
# -------------------------
def normalize_features(df):
    df = df.copy()
    df['Pt'] /= df['Pt'].max()
    df['FPR'] /= df['FPR'].max()
    df['GCT'] /= df['GCT'].max()
    df['Cadence'] /= df['Cadence'].max()
    df['Grade'] /= df['Grade'].abs().max() if df['Grade'].abs().max() > 0 else 1
    return df

# -------------------------
# OBSERVATION EQUATION (FIXED)
# -------------------------
def speed_observation(state, Pt, FPRt, grade_t, Cadence_t,
                      k0, k2, k3, k4, lam, b):

    effective_energy = state.E * (1 - state.F)

    speed = (
        Pt * (1 + effective_energy)**lam
    ) / (
        k0
        + k2 * FPRt
        + k3 * abs(grade_t)
        + k4 * Cadence_t
    )

    return speed + b 


# -------------------------
# PREPROCESS STRYD DATA
# -------------------------
def preprocess_stryd(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Power (w/kg)'] > 0].copy()

    df['FPR'] = df['Form Power (w/kg)'] / df['Power (w/kg)']

    elevation_diff = df['Watch Elevation (m)'].diff().fillna(0)
    horizontal_diff = df['Watch Distance (meters)'].diff().replace(0, np.nan)
    df['Grade'] = (elevation_diff / horizontal_diff).fillna(0)

    df['Pt'] = df['Power (w/kg)']
    df['GCT'] = df['Ground Time (ms)'] / 1000.0
    df['Cadence'] = df['Cadence (spm)']

    df = normalize_features(df)
    return df

# -------------------------
# SIMULATION LOOP
# -------------------------
def simulate_run(df, k0, k2, k3, k4, lam, b):
    states = [RunnerState()]
    speeds = []

    for i in range(len(df)):
        row = df.iloc[i]
        state = states[-1]

        speed = speed_observation(
            state,
            row['Pt'], row['FPR'], row['Grade'], row['Cadence'],
            k0, k2, k3, k4, lam, b
        )
        speeds.append(speed)

        states.append(
            update_state(
                state,
                row['Pt'], row['FPR'], row['GCT'], row['Grade']
            )
        )

    return speeds, states

# -------------------------
# LOSS FUNCTION
# -------------------------
def loss(params, df):
    k0, k2, k3, k4, lam, b = params
    pred, _ = simulate_run(df, k0, k2, k3, k4, lam, b)
    return np.mean((np.array(pred) - df['Watch Speed (m/s)'].values) ** 2)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    df_train = preprocess_stryd("Run2.csv")

    init = [kappa0, kappa2, kappa3, kappa4, lam, speed_bias]
    bounds = [
    (1e-3, None),  # k0
    (0, None),     # k2
    (0, None),     # k3
    (0, None),     # k4
    (0.5, 3.0),    # lam
    (-2.0, 2.0)    # speed bias (m/s)
    ]


    res = minimize(loss, init, args=(df_train,), bounds=bounds)
    k0, k2, k3, k4, lam, b = res.x
    print("Trained params:", res.x)

    df_test = preprocess_stryd("Run3.csv")
    pred, states = simulate_run(df_test, k0, k2, k3, k4, lam, b)

    plt.plot(df_test['Watch Speed (m/s)'], label="Observed")
    plt.plot(pred, label="Predicted")
    plt.legend()
    plt.show()
