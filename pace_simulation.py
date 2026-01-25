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
beta = 0.06

gamma = 0.002
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
# OBSERVATION EQUATION
# -------------------------
def speed_observation(state, Pt, FPRt, grade_t, Cadence_t,
                      k0, k2, k3, k4, lam, b):
    # fatigue reduces how effectively energy is used
    effective_energy = state.E * (1 - state.F)

    speed = (
        Pt * (1 + effective_energy) ** lam
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
# MULTI-RUN LOSS FUNCTION
# -------------------------
def loss(params, dfs):
    k0, k2, k3, k4, lam, b = params
    run_losses = []

    for df in dfs:
        pred, _ = simulate_run(df, k0, k2, k3, k4, lam, b)
        err = np.mean(
            (np.array(pred) - df['Watch Speed (m/s)'].values) ** 2
        )
        run_losses.append(err)

    return np.mean(run_losses)

# -------------------------
# FORECAST / SIMULATION (like GFS)
# -------------------------
def forecast_run(initial_state, n_steps, k0, k2, k3, k4, lam, b,
                 Pt_series=None, FPR_series=None, GCT_series=None, Grade_series=None, Cadence_series=None):
    """
    Propagate the runner state forward for n_steps using optionally given input series.
    If no input series are provided, assume Pt=1, FPR=0, GCT=0, Grade=0, Cadence=1.
    """

    states = [initial_state]
    speeds = []

    for t in range(n_steps):
        state = states[-1]

        Pt = Pt_series[t] if Pt_series is not None else 1.0
        FPRt = FPR_series[t] if FPR_series is not None else 0.0
        GCTt = GCT_series[t] if GCT_series is not None else 0.0
        grade_t = Grade_series[t] if Grade_series is not None else 0.0
        cadence_t = Cadence_series[t] if Cadence_series is not None else 1.0

        speed = speed_observation(state, Pt, FPRt, grade_t, cadence_t, k0, k2, k3, k4, lam, b)
        speeds.append(speed)

        next_state = update_state(state, Pt, FPRt, GCTt, grade_t)
        states.append(next_state)

    return speeds, states

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    # -------- TRAINING RUNS --------
    train_files = [
        "Run1.csv",
        "Run2.csv",
        "Run3.csv",
        "Run5.csv", 
        "Run6.csv"
    ]

    dfs_train = [preprocess_stryd(f) for f in train_files]

    init = [kappa0, kappa2, kappa3, kappa4, lam, speed_bias]

    bounds = [
        (1e-3, None),   # k0
        (1e-3, None),   # k2
        (1e-3, None),   # k3
        (1e-3, None),   # k4
        (0.5, 3.0),     # lam
        (-2.0, 2.0)     # speed bias
    ]

    res = minimize(loss, init, args=(dfs_train,), bounds=bounds)
    k0, k2, k3, k4, lam, b = res.x
    print("Trained params:", res.x)

    # -------- TEST RUN --------
    df_test = preprocess_stryd("Run4.csv")
    pred, states = simulate_run(df_test, k0, k2, k3, k4, lam, b)

    plt.figure()
    plt.plot(df_test['Watch Speed (m/s)'], label="Observed")
    plt.plot(pred, label="Predicted")
    plt.xlabel("Time step")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.title("Predicted vs Observed Speed")
    plt.show()

    # -------- FORECAST / SIMULATION EXAMPLE --------
    initial_state = RunnerState(fatigue=states[-1].F, energy=states[-1].E)  # start from end of test run
    n_forecast_steps = len(df_test)  # forecast same length as test run

    # use test run features for forecast (realistic forecast)
    speeds_forecast, states_forecast = forecast_run(
        initial_state,
        n_forecast_steps,
        k0, k2, k3, k4, lam, b,
        Pt_series=df_test['Pt'].values,
        FPR_series=df_test['FPR'].values,
        GCT_series=df_test['GCT'].values,
        Grade_series=df_test['Grade'].values,
        Cadence_series=df_test['Cadence'].values
    )

    times = np.arange(n_forecast_steps)
    F = [s.F for s in states_forecast[:-1]]
    E = [s.E for s in states_forecast[:-1]]

    fig, ax1 = plt.subplots()
    ax1.plot(times, speeds_forecast, 'r-', label='Forecast Speed')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Speed (m/s)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.plot(times, F, 'b--', label='Fatigue')
    ax2.plot(times, E, 'g-.', label='Energy')
    ax2.set_ylabel('State', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Forecast Simulation")
    plt.show()
