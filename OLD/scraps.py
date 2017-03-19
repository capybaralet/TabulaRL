

#----------------------------------------
def q_sigma(policy, Q, rollouts, sigma_fn):
    """
    Perform Q_sigma on a trajectory.

    Assumes that the target policy is greedy wrt Q

    -----------------------
    policy: Q_vals --> P(action)
    Q: state --> Q_vals
    sigma_fn: rho --> P(sigma=1)

    sigma:
        0 - use tree backup
        1 - use IS
    """

    return


#----------------------------------------
# sigma functions
def sigma1(rho):
    return rho < 1

#----------------------------------------
# TODO: do this efficiently for an entire rollout...
def nstep_return(Q, gamma, rollout, n, t):
    """aka 'G' on page 13 of https://webdocs.cs.ualberta.ca/~sutton/609%20dropbox/slides%20(pdf%20and%20keynote)/13-multistep.pdf """
    states, actions, rewards = rollout
    assert len(states) == len(actions) == len(rewards)

    discounted_rewards = rewards * gamma ** (np.arange(len(rewards)))

    if len(states) < n:
        return sum(discounted_rewards)
    else:
        return sum(discounted_rewards) + Q(states[n+1], actions[n+1])



def expected_returns(Q, s, policy):
    """ expected returns acting according to policy(Q) in state s """
    Q_vals = Q[s]
    P_a = policy(Q_vals)
    return (P_a * Q_vals).sum()

