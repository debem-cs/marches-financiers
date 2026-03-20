"""
Optimal Equity Injection for Banking Networks
=============================================
Based on: Bayraktar, Guo, Tang, Zhang (2025)
"Systemic robustness: a mean-field particle system approach"

Implements regression-based dynamic programming (fitted value iteration)
for both:
  - Case 1: Independent banks
  - Case 2: Interconnected banks (default cascades)

Two objectives:
  - U: Maximize expected number of survivors
  - V: Maximize probability all banks survive
"""

import numpy as np
from itertools import product as cartesian_product
from scipy.linalg import svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. PARAMETERS
# ============================================================
class Params:
    N = 3           # Number of banks
    T = 5           # Time horizon
    c = 1.0         # Total budget per period
    S0 = 2.0        # Initial capital per bank (all start equal)
    mu = 0.0        # Mean of shock distribution
    sigma = 1.0     # Std of shock distribution
    beta = 1.5      # Interbank liability (Case 2)
    
    # Algorithm parameters
    N_train = 200   # Training states per time step
    M = 80          # Monte Carlo samples for target computation
    lam = 0.01      # Ridge regularization
    
    # Action discretization
    n_action_levels = 3


# ============================================================
# 2. ACTION SPACE DISCRETIZATION
# ============================================================
def build_action_space(N, c, n_levels=3):
    """
    Build discrete action space: all allocations (b1,...,bN) with
    bi >= 0 and sum(bi) <= c.
    
    For tractability, we use a coarse grid.
    """
    # For small N, enumerate simplex points
    levels = np.linspace(0, c, n_levels)
    actions = []
    for combo in cartesian_product(levels, repeat=N):
        a = np.array(combo)
        if np.sum(a) <= c + 1e-10:
            actions.append(a)
    return np.array(actions)


def build_action_space_smart(N, c, n_alloc=5):
    """
    Smarter action space: allocate budget to the weakest banks.
    Actions: 'give c to bank i', 'split c among k weakest', 'do nothing', etc.
    This returns action *rules* as functions.
    """
    actions = []
    # Action 0: Do nothing
    actions.append(np.zeros(N))
    # Action: give all to bank i
    for i in range(N):
        a = np.zeros(N)
        a[i] = c
        actions.append(a)
    # Action: split equally among all
    a = np.ones(N) * c / N
    actions.append(a)
    # Action: split among 2 weakest (placeholder - actual depends on state)
    # For now, we use static actions. State-dependent actions handled in target computation.
    return np.array(actions)


# ============================================================
# 3. FEATURE FUNCTIONS
# ============================================================
def compute_features(x, e):
    """
    Compute feature vector phi(x, e) for state (x, e).
    
    x: (N,) capital levels
    e: (N,) survival indicators {0,1}
    
    Returns: (P,) feature vector
    """
    N = len(x)
    n_alive = np.sum(e)
    frac_alive = n_alive / N
    
    avg_capital = np.mean(x)
    
    # Average capital among survivors
    if n_alive > 0:
        avg_capital_survivors = np.sum(e * x) / n_alive
        min_capital_survivors = np.min(x[e > 0]) if n_alive > 0 else 0.0
        var_capital_survivors = np.var(x[e > 0]) if n_alive > 1 else 0.0
        max_capital_survivors = np.max(x[e > 0]) if n_alive > 0 else 0.0
    else:
        avg_capital_survivors = 0.0
        min_capital_survivors = 0.0
        var_capital_survivors = 0.0
        max_capital_survivors = 0.0
    
    features = np.array([
        1.0,                          # Intercept
        frac_alive,                   # Fraction alive
        avg_capital,                  # Average capital (all)
        avg_capital_survivors,        # Average capital (survivors)
        min_capital_survivors,        # Weakest survivor
        max_capital_survivors,        # Strongest survivor
        avg_capital ** 2,             # Quadratic: avg capital squared
        var_capital_survivors,        # Variance among survivors
        frac_alive * avg_capital_survivors,  # Interaction
        min_capital_survivors ** 2,   # Quadratic: min capital
        frac_alive ** 2,             # Quadratic: fraction alive
    ])
    return features

P_FEATURES = 11  # Number of features


# ============================================================
# 4. DEFAULT CASCADE (Case 2)
# ============================================================
def apply_default_cascade(S, e, beta, N):
    """
    Iterative default cascade: when a bank defaults, survivors lose beta/N.
    Repeat until no new defaults.
    
    Returns updated (S, e).
    """
    S = S.copy()
    e = e.copy()
    max_iter = N + 1  # At most N rounds of cascading
    for _ in range(max_iter):
        e_new = e * (S > 0).astype(float)
        D = np.sum(e - e_new)  # Number of new defaults
        if D == 0:
            break
        loss = beta * D / N
        # Surviving banks suffer the loss
        S[e_new > 0] -= loss
        e = e_new
    # Final update
    e = e * (S > 0).astype(float)
    return S, e


# ============================================================
# 5. STATE DYNAMICS
# ============================================================
def step_independent(x, e, shock, action):
    """Case 1: Independent banks - one step dynamics."""
    x_new = x + shock + action
    e_new = e * (x_new > 0).astype(float)
    return x_new, e_new


def step_interconnected(x, e, shock, action, beta, N):
    """Case 2: Interconnected banks - one step with cascade."""
    x_new = x + shock + action
    x_new, e_new = apply_default_cascade(x_new, e, beta, N)
    return x_new, e_new


# ============================================================
# 6. STATE-DEPENDENT ACTIONS
# ============================================================
def generate_state_dependent_actions(x, e, c, N):
    """
    Generate sensible actions given the current state.
    Focus budget on banks that need it most.
    """
    actions = []
    alive_mask = e > 0
    n_alive = int(np.sum(alive_mask))
    
    # Action 0: Do nothing
    actions.append(np.zeros(N))
    
    if n_alive == 0:
        return np.array(actions)
    
    # Socialistic policy: split equally among survivors
    a = np.zeros(N)
    a[alive_mask] = c / n_alive
    actions.append(a)
    
    # Give all to weakest survivor
    if n_alive > 0:
        alive_indices = np.where(alive_mask)[0]
        weakest = alive_indices[np.argmin(x[alive_indices])]
        a = np.zeros(N)
        a[weakest] = c
        actions.append(a)
    
    # Give all to each individual survivor
    for i in range(N):
        if e[i] > 0:
            a = np.zeros(N)
            a[i] = c
            actions.append(a)
    
    # Split among 2 weakest survivors
    if n_alive >= 2:
        alive_indices = np.where(alive_mask)[0]
        sorted_by_capital = alive_indices[np.argsort(x[alive_indices])]
        two_weakest = sorted_by_capital[:2]
        a = np.zeros(N)
        a[two_weakest] = c / 2
        actions.append(a)
    
    # Split among 3 weakest
    if n_alive >= 3:
        three_weakest = sorted_by_capital[:3]
        a = np.zeros(N)
        a[three_weakest] = c / 3
        actions.append(a)
    
    return np.array(actions)


# ============================================================
# 7. ROLLOUT POLICY (for generating training states)
# ============================================================
def rollout_policy(x, e, c, N):
    """Simple rollout: split budget equally among survivors."""
    a = np.zeros(N)
    alive = e > 0
    n_alive = np.sum(alive)
    if n_alive > 0:
        a[alive] = c / n_alive
    return a


# ============================================================
# 8. TERMINAL VALUE
# ============================================================
def terminal_value_U(x, e):
    """Objective U: number of survivors."""
    return np.sum(e * (x > 0))


def terminal_value_V(x, e):
    """Objective V: all survive indicator."""
    return np.prod(e * (x > 0))


# ============================================================
# 9. RIDGE REGRESSION
# ============================================================
def ridge_regression(Phi, Y, lam):
    """
    Solve theta = (Phi^T Phi + lambda I)^{-1} Phi^T Y
    Using SVD for numerical stability.
    """
    U, s, Vt = svd(Phi, full_matrices=False)
    # theta = V diag(s/(s^2+lambda)) U^T Y
    d = s / (s**2 + lam)
    theta = Vt.T @ np.diag(d) @ U.T @ Y
    return theta


# ============================================================
# 10. MAIN ALGORITHM: Regression-Based Dynamic Programming
# ============================================================
def run_algorithm(params, objective='U', case=1, verbose=True):
    """
    Full backward induction algorithm.
    
    Args:
        params: Params object
        objective: 'U' or 'V'
        case: 1 (independent) or 2 (interconnected)
        verbose: print progress
    
    Returns:
        thetas: dict t -> theta_t (weight vectors)
        diagnostics: dict with training info
    """
    N = params.N
    T = params.T
    c = params.c
    S0 = params.S0
    sigma = params.sigma
    mu = params.mu
    beta = params.beta
    N_train = params.N_train
    M = params.M
    lam = params.lam
    
    terminal_fn = terminal_value_U if objective == 'U' else terminal_value_V
    step_fn = step_independent if case == 1 else (
        lambda x, e, shock, action: step_interconnected(x, e, shock, action, beta, N)
    )
    
    thetas = {}
    diagnostics = {'bellman_errors': [], 'r_squared': []}
    
    # -------------------------------------------------------
    # Step 0: Terminal time T - fit theta_T
    # -------------------------------------------------------
    if verbose:
        print(f"=== Running: Objective {objective}, Case {case} ===")
        print(f"  N={N}, T={T}, c={c}, S0={S0}, sigma={sigma}")
        if case == 2:
            print(f"  beta={beta}")
        print(f"  N_train={N_train}, M={M}, lambda={lam}")
        print()
    
    # Generate terminal states by simulating to time T
    Phi_T = np.zeros((N_train, P_FEATURES))
    Y_T = np.zeros(N_train)
    
    for n in range(N_train):
        x = np.ones(N) * S0
        e = np.ones(N)
        for t in range(T):
            shock = np.random.normal(mu, sigma, N)
            action = rollout_policy(x, e, c, N)
            x, e = step_fn(x, e, shock, action)
        Phi_T[n] = compute_features(x, e)
        Y_T[n] = terminal_fn(x, e)
    
    thetas[T] = ridge_regression(Phi_T, Y_T, lam)
    
    if verbose:
        pred = Phi_T @ thetas[T]
        r2 = 1 - np.sum((Y_T - pred)**2) / (np.sum((Y_T - np.mean(Y_T))**2) + 1e-10)
        print(f"  t={T} (terminal): R² = {r2:.4f}")
    
    # -------------------------------------------------------
    # Backward Iteration: t = T-1, T-2, ..., 0
    # -------------------------------------------------------
    for t in range(T - 1, -1, -1):
        Phi_t = np.zeros((N_train, P_FEATURES))
        Y_t = np.zeros(N_train)
        
        for n in range(N_train):
            # Step 1: Generate training state at time t
            x = np.ones(N) * S0
            e = np.ones(N)
            for s in range(t):
                shock = np.random.normal(mu, sigma, N)
                action = rollout_policy(x, e, c, N)
                x, e = step_fn(x, e, shock, action)
            
            # Step 2: Compute target via Monte Carlo
            actions = generate_state_dependent_actions(x, e, c, N)
            best_Q = -np.inf
            
            for a in actions:
                Q_a = 0.0
                for m in range(M):
                    shock = np.random.normal(mu, sigma, N)
                    x_next, e_next = step_fn(x, e, shock, a)
                    
                    if t + 1 == T:
                        Q_a += terminal_fn(x_next, e_next)
                    else:
                        phi_next = compute_features(x_next, e_next)
                        Q_a += np.dot(thetas[t + 1], phi_next)
                
                Q_a /= M
                if Q_a > best_Q:
                    best_Q = Q_a
            
            Phi_t[n] = compute_features(x, e)
            Y_t[n] = best_Q
        
        # Step 3: Ridge regression
        thetas[t] = ridge_regression(Phi_t, Y_t, lam)
        
        if verbose:
            pred = Phi_t @ thetas[t]
            ss_res = np.sum((Y_t - pred)**2)
            ss_tot = np.sum((Y_t - np.mean(Y_t))**2) + 1e-10
            r2 = 1 - ss_res / ss_tot
            diagnostics['r_squared'].append(r2)
            print(f"  t={t}: R² = {r2:.4f}, mean target = {np.mean(Y_t):.4f}")
    
    if verbose:
        print("\nDone!")
    
    return thetas, diagnostics


# ============================================================
# 11. POLICY EVALUATION via Monte Carlo Simulation
# ============================================================
def evaluate_policy(thetas, params, objective='U', case=1, 
                    policy_type='optimal', n_sim=800):
    """
    Evaluate a policy by Monte Carlo simulation.
    
    policy_type:
      'optimal' - use learned value function to pick best action
      'socialistic' - split equally among survivors
      'weakest' - give all to weakest survivor
      'nothing' - no intervention
      'strongest' - give all to strongest survivor
    """
    N = params.N
    T = params.T
    c = params.c
    S0 = params.S0
    sigma = params.sigma
    mu = params.mu
    beta = params.beta
    
    terminal_fn = terminal_value_U if objective == 'U' else terminal_value_V
    step_fn = step_independent if case == 1 else (
        lambda x, e, shock, action: step_interconnected(x, e, shock, action, beta, N)
    )
    
    results = np.zeros(n_sim)
    survival_paths = np.zeros((n_sim, T + 1))
    
    for sim in range(n_sim):
        x = np.ones(N) * S0
        e = np.ones(N)
        survival_paths[sim, 0] = np.sum(e)
        
        for t in range(T):
            # Choose action based on policy
            if policy_type == 'nothing':
                action = np.zeros(N)
            elif policy_type == 'socialistic':
                action = np.zeros(N)
                alive = e > 0
                n_alive = np.sum(alive)
                if n_alive > 0:
                    action[alive] = c / n_alive
            elif policy_type == 'weakest':
                action = np.zeros(N)
                alive = np.where(e > 0)[0]
                if len(alive) > 0:
                    weakest = alive[np.argmin(x[alive])]
                    action[weakest] = c
            elif policy_type == 'strongest':
                action = np.zeros(N)
                alive = np.where(e > 0)[0]
                if len(alive) > 0:
                    strongest = alive[np.argmax(x[alive])]
                    action[strongest] = c
            elif policy_type == 'optimal':
                # Use learned thetas to pick best action
                actions = generate_state_dependent_actions(x, e, c, N)
                best_val = -np.inf
                best_action = np.zeros(N)
                
                M_eval = 30  # Fewer MC samples for speed
                for a in actions:
                    Q_a = 0.0
                    for m in range(M_eval):
                        shock = np.random.normal(mu, sigma, N)
                        x_next, e_next = step_fn(x, e, shock, a)
                        if t + 1 == T:
                            Q_a += terminal_fn(x_next, e_next)
                        elif t + 1 in thetas:
                            phi_next = compute_features(x_next, e_next)
                            Q_a += np.dot(thetas[t + 1], phi_next)
                    Q_a /= M_eval
                    if Q_a > best_val:
                        best_val = Q_a
                        best_action = a
                action = best_action
            
            shock = np.random.normal(mu, sigma, N)
            x, e = step_fn(x, e, shock, action)
            survival_paths[sim, t + 1] = np.sum(e)
        
        results[sim] = terminal_fn(x, e)
    
    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'survival_paths': survival_paths,
        'results': results
    }


# ============================================================
# 12. COMPARISON PLOTS
# ============================================================
def plot_policy_comparison(params, objective='U', case=1, thetas=None,
                          save_prefix='plot'):
    """Generate comparison plots for different policies."""
    
    policies = ['nothing', 'weakest', 'strongest', 'socialistic']
    if thetas is not None:
        policies.append('optimal')
    
    policy_labels = {
        'nothing': 'No Intervention',
        'weakest': 'Inject Weakest',
        'strongest': 'Inject Strongest', 
        'socialistic': 'Socialistic (Equal Split)',
        'optimal': 'Optimal (Learned)'
    }
    
    colors = {
        'nothing': '#d62728',
        'weakest': '#ff7f0e',
        'strongest': '#9467bd',
        'socialistic': '#2ca02c',
        'optimal': '#1f77b4'
    }
    
    results = {}
    print(f"\nEvaluating policies (Objective {objective}, Case {case})...")
    for p in policies:
        print(f"  {policy_labels[p]}...", end=' ')
        r = evaluate_policy(thetas if thetas else {}, params, objective, case, p, n_sim=800)
        results[p] = r
        print(f"E[{objective}] = {r['mean']:.4f} ± {r['std']:.4f}")
    
    case_label = "Independent" if case == 1 else "Interconnected (Cascades)"
    obj_label = "E[# Survivors]" if objective == 'U' else "P(All Survive)"
    
    # --- Plot 1: Bar chart of expected values ---
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [policy_labels[p] for p in policies]
    means = [results[p]['mean'] for p in policies]
    stds = [results[p]['std'] / np.sqrt(1500) for p in policies]  # Standard error
    cols = [colors[p] for p in policies]
    
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=cols, edgecolor='black', alpha=0.85)
    ax.set_ylabel(obj_label, fontsize=13)
    ax.set_title(f'Policy Comparison — {case_label}\n(N={params.N}, T={params.T}, c={params.c}, σ={params.sigma})',
                 fontsize=14)
    ax.set_ylim(bottom=0)
    
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{m:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    fname1 = f'{save_prefix}_bar_{objective}_case{case}.png'
    plt.savefig(fname1, dpi=150)
    plt.close()
    
    # --- Plot 2: Mean survival paths over time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in policies:
        mean_path = np.mean(results[p]['survival_paths'], axis=0)
        ax.plot(range(params.T + 1), mean_path, label=policy_labels[p],
                color=colors[p], linewidth=2.5)
    
    ax.set_xlabel('Time Step', fontsize=13)
    ax.set_ylabel('Mean Number of Surviving Banks', fontsize=13)
    ax.set_title(f'Survival Over Time — {case_label}\n(N={params.N}, T={params.T})',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, params.N + 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname2 = f'{save_prefix}_paths_{objective}_case{case}.png'
    plt.savefig(fname2, dpi=150)
    plt.close()
    
    return results, fname1, fname2


def plot_budget_sensitivity(params, objective='U', case=1, save_prefix='plot'):
    """Show how budget c affects outcomes under socialistic policy."""
    budgets = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    means = []
    
    for c_val in budgets:
        p = Params()
        p.__dict__.update(params.__dict__)
        p.c = c_val
        r = evaluate_policy({}, p, objective, case, 'socialistic', n_sim=500)
        means.append(r['mean'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(budgets, means, 'o-', color='#2ca02c', linewidth=2.5, markersize=8)
    ax.set_xlabel('Budget c', fontsize=13)
    obj_label = "E[# Survivors]" if objective == 'U' else "P(All Survive)"
    ax.set_ylabel(obj_label, fontsize=13)
    case_label = "Independent" if case == 1 else "Interconnected"
    ax.set_title(f'Budget Sensitivity — Socialistic Policy ({case_label})', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'{save_prefix}_budget_{objective}_case{case}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_cascade_effect(params, save_prefix='plot'):
    """Compare Case 1 vs Case 2 under different policies."""
    policies = ['nothing', 'socialistic']
    policy_labels = {'nothing': 'No Intervention', 'socialistic': 'Socialistic'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, obj in enumerate(['U', 'V']):
        ax = axes[idx]
        obj_label = "E[# Survivors]" if obj == 'U' else "P(All Survive)"
        
        x_pos = np.arange(len(policies))
        width = 0.35
        
        means_c1 = []
        means_c2 = []
        for p in policies:
            r1 = evaluate_policy({}, params, obj, 1, p, n_sim=500)
            r2 = evaluate_policy({}, params, obj, 2, p, n_sim=500)
            means_c1.append(r1['mean'])
            means_c2.append(r2['mean'])
        
        bars1 = ax.bar(x_pos - width/2, means_c1, width, label='Independent (Case 1)',
                       color='#1f77b4', edgecolor='black', alpha=0.85)
        bars2 = ax.bar(x_pos + width/2, means_c2, width, label='Cascades (Case 2)',
                       color='#d62728', edgecolor='black', alpha=0.85)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([policy_labels[p] for p in policies])
        ax.set_ylabel(obj_label, fontsize=12)
        ax.set_title(f'Objective {obj}: Impact of Default Cascades', fontsize=13)
        ax.legend()
        ax.set_ylim(bottom=0)
        
        for bar, m in zip(list(bars1) + list(bars2), means_c1 + means_c2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fname = f'{save_prefix}_cascade_comparison.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ============================================================
# 13. MAIN: Run everything
# ============================================================
def main():
    params = Params()
    output_dir = '/home//results'
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, 'equity')
    
    print("=" * 60)
    print("OPTIMAL EQUITY INJECTION - FULL ANALYSIS")
    print("=" * 60)
    
    # --- A. Run algorithm for each configuration ---
    configs = [
        ('U', 1), ('U', 2),  # Focus on objective U for both cases
    ]
    
    all_thetas = {}
    for obj, case in configs:
        print(f"\n{'='*50}")
        thetas, diag = run_algorithm(params, objective=obj, case=case, verbose=True)
        all_thetas[(obj, case)] = thetas
    
    # --- B. Policy comparison plots ---
    print(f"\n{'='*50}")
    print("POLICY EVALUATION")
    print("=" * 50)
    
    all_figs = []
    for obj, case in configs:
        thetas = all_thetas[(obj, case)]
        results, f1, f2 = plot_policy_comparison(
            params, objective=obj, case=case, thetas=thetas, save_prefix=prefix
        )
        all_figs.extend([f1, f2])
    
    # --- C. Budget sensitivity (Case 1 only) ---
    print("\n--- Budget Sensitivity ---")
    f = plot_budget_sensitivity(params, 'U', 1, prefix)
    all_figs.append(f)
    
    # --- D. Cascade comparison ---
    print("\n--- Cascade Effect Analysis ---")
    f = plot_cascade_effect(params, prefix)
    all_figs.append(f)
    
    print(f"\nAll figures saved in: {output_dir}")
    return all_figs


if __name__ == '__main__':
    figs = main()
