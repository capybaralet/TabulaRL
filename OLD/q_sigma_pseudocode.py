
# Initialize Q(s, a) arbitrarily, ∀s ∈ S, a ∈ A
Q = np.zeros((grid_width**2, 4))
# TODO: Initialize π to be ε-greedy with respect to Q, or as a fixed given policy
#
# Parameters: step size α ∈ (0, 1], small ε > 0, a positive integer n
# TODO: All store and access operations can take their index mod n
n = backup_length

S_t = np.empty(backup_length)
A_t = np.empty(barkup_length)
R_t = np.empty(barkup_length)
sigma_t = np.empty(backup_length)
Q_t = np.empty(backup_length)
delta_t = np.empty(backup_length)
pi_t = np.empty(backup_length)
rho_t = np.empty(backup_length)

#Repeat (for each episode):
for episode in range(num_episodes):
    # Initialize and store S0 != terminal
    s = 0
    S_t[0] = s
    # Select and store an action A0 ∼ π(·|S0)
    a = mu.sample(s)
    A_t[0] = a
    # Store Q(S0, A0) as Q0
    Q_t[0] = Q[s,a]
    # T ← ∞
    TT = np.inf
    #For t = 0, 1, 2, . . . :
    tt = 0
    finished = 0
    while not finished:
        tt += 1
        # If t < T:
        if tt < TT:
            # Take action At
            s = next_state(s, a)
            is_terminal = (new_s == grid_width**2-1)
            # Observe the next reward R
            R_t[tt] = - is_terminal
            # Observe and store the next state as St+1
            S_t[tt] = s
            # If St+1 is terminal:
            if is_terminal:
                # T ← t + 1
                TT = tt + 1
                # Store R − Qt as δt
                delta_t[tt] = -is_terminal - Q_t[tt]
            # else:
            else:
                # Select and store an action At+1 ∼ µ(·|St+1)
                a = mu.sample(s)
                A_t[tt] = a
                # Store π(At+1|St+1) as πt+1
                pi_t[tt] = pi.P_a(Q[s])
                # Store π(At+1|St+1) / µ(At+1|St+1) as ρt+1
                rho_t[tt] = pi_t[tt][a] /  mu.P_a(Q[s])[a]
                # Select and store σt+1
                sigma_t[tt] = sigma_fn(rho_t)
                # Store Q(St+1, At+1) as Qt+1
                Q_t[tt] = Q[s,a]
                # Store R + γσt+1Qt+1 + γ(1 − σt+1) SUM_a π(a|St+1)Q(St+1, a) − Qt as δt
                delta_t[tt] = -is_terminal + sigma_t[tt]*Q_t[tt] + (1-sigma_t[tt])*np.sum(pi_t[tt] * Q_t[tt])
        # τ ← t − n + 1 (τ is the time whose estimate is being updated)
        tau = tt - n + 1
        # If τ ≥ 0:
        if tau >= 0:
            # ρ ← 1
            rho = 1
            # E ← 1
            E = 1
            # G ← Qτ
            G = Q_t[tau]
            # For k = τ, . . . , min(τ + n − 1, T − 1):
            for k in range(tau, min(tau + n - 1, TT - 1)):
                # G ← G + Eδk
                G += E * delta_t[k]
                # E ← γE (1 − σk+1)πk+1 + σk+1
                E *= (1 - sigma_t[k+1] * pi_t[k][A_t[k]]) + sigma_t[k+1]
                # ρ ← ρ(1 − σk + σkρk)
                rho *= (1 - sigma_t[k] + sigma_t[k] * rho_t[k])
            # Q(Sτ , Aτ ) ← Q(Sτ , Aτ ) + αρ [G − Q(Sτ , Aτ )]
            S_tau, A_tau = S_t[tau], A_t[tau]
            Q[S_tau, A_tau] += lr * rho * (G - Q[S_tau, A_tau])
        #Until τ = T − 1
        if tau == TT - 1:
            finished = 1
