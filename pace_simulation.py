# pace_simulation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS (tune these)
# -------------------------
alpha = 0.01    # fatigue from power
delta = 0.005   # fatigue from FPR
epsilon = 0.002 # fatigue from GCT
beta = 0.05     # natural recovery of fatigue

gamma = 0.02    # energy cost from power
eta = 0.1       # energy cost sensitivity to grade

kappa0 = 5.0   # base pace (min/mile)
kappa1 = 0.1   # fatigue coefficient
kappa2 = 0.05  # FPR coefficient
kappa3 = 0.2   # grade coefficient
lam = 1.0      # energy exponent

dt = 1.0       # time step, in seconds

# -------------------------
# STATE VECTOR
# -------------------------
class RunnerState:
    def __init__(self, fatigue=0.0, energy=1.0):
        self.F = fatigue  # Fatigue
        self.E = energy   # Expendable energy (fraction of total)

# -------------------------
# DISCRETIZED ODE UPDATE
# -------------------------
def update_state(state, Pt, FPRt, GCTt, grade_t):
    """Evolve state vector one timestep."""
    # Fatigue
    F_next = state.F + dt * (alpha*Pt + delta*FPRt + epsilon*GCTt - beta*state.F)
    
    # Expendable Energy
    E_next = state.E + dt * (-gamma*Pt*(1 + eta*abs(grade_t)))
    
    # Keep energy within [0,1]
    E_next = max(min(E_next, 1.0), 0.0)
    
    return RunnerState(fatigue=F_next, energy=E_next)

# -------------------------
# OBSERVATION EQUATION
# -------------------------
def pace_observation(state, Pt, FPRt, grade_t):
    """Compute pace from state vector and inputs."""
    pace = (kappa0 + kappa1*state.F + kappa2*FPRt + kappa3*abs(grade_t)) / (Pt * state.E**lam)
    return pace

# -------------------------
# SIMULATION LOOP
# -------------------------
def simulate_run(data):
    """
    data: pandas DataFrame with columns ['Power', 'FPR', 'GCT', 'Grade']
    """
    n_steps = len(data)
    states = [RunnerState()]
    paces = []

    for i in range(n_steps):
        Pt = data.loc[i, 'Power']
        FPRt = data.loc[i, 'FPR']
        GCTt = data.loc[i, 'GCT']
        grade_t = data.loc[i, 'Grade']

        # Compute pace from previous state
        pace = pace_observation(states[-1], Pt, FPRt, grade_t)
        paces.append(pace)

        # Update state
        next_state = update_state(states[-1], Pt, FPRt, GCTt, grade_t)
        states.append(next_state)

    return paces, states[1:]  # skip initial state in output

# -------------------------
# VISUALIZATION
# -------------------------
def plot_results(paces, states):
    times = np.arange(len(paces))
    F = [s.F for s in states]
    E = [s.E for s in states]

    fig, ax1 = plt.subplots()

    ax1.plot(times, paces, 'r-', label='Pace')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Pace', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.plot(times, F, 'b--', label='Fatigue')
    ax2.plot(times, E, 'g-.', label='Energy')
    ax2.set_ylabel('State', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    fig.tight_layout()
    plt.legend()
    plt.show()

# -------------------------
# MAIN FUNCTION (example)
# -------------------------
if __name__ == "__main__":
    # Example data: replace with Stryd pod CSV
    example_data = pd.DataFrame({
        'Power': np.random.normal(250, 10, 300),
        'FPR': np.random.normal(0.25, 0.02, 300),
        'GCT': np.random.normal(0.25, 0.01, 300),
        'Grade': np.zeros(300)
    })

    paces, states = simulate_run(example_data)
    plot_results(paces, states)
