'''
Finite horizon tabular agents.

This is a collection of some of the classic benchmark algorithms for efficient
reinforcement learning in a tabular MDP with little/no prior knowledge.
We provide implementations of:

- PSRL
- Gaussian PSRL
- UCBVI
- BEB
- BOLT
- UCRL2
- Epsilon-greedy

author: iosband@stanford.edu
'''

import numpy as np
from agent import *

from query_functions import AlwaysQuery


def modifyPrior(prior): 
    def nonStayKnown(sa, p ): 
        #non 'stay' actions have infinite precision
        _, action = sa
        mu, tau = p

        if action != 0: 
            return (mu, 1e10)
        else: 
            return (mu, tau)

    return { k : nonStayKnown(k, v) for k,v in prior.iteritems() } 


class FiniteHorizonTabularAgent(FiniteHorizonAgent):
    '''
    Simple tabular Bayesian learner from Tabula Rasa.

    Child agents will mainly implement:
        update_policy

    Important internal representation is given by qVals and qMax.
        qVals - qVals[state, timestep] is vector of Q values for each action
        qMax - qMax[timestep] is the vector of optimal values at timestep

    '''

    
    # FIXME: P_true, R_true need to be accounted for (everywhere!) - (NTS: I don't know if there are any specific places)
    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., 
                 P_true=None, R_true=None, query_function=AlwaysQuery(0.), stop_learning=True, 
                 reward_depends_on_action=True,
                 **kwargs):
        '''
        Tabular episodic learner for time-homoegenous MDP.
        Must be used together with true state feature extractor.

        Args:
            nState - int - number of states
            nAction - int - number of actions
            alpha0 - prior weight for uniform Dirichlet
            mu0 - prior mean rewards
            tau0 - precision of prior mean rewards
            tau - precision of reward noise

        Returns:
            tabular learner, to be inherited from
        '''
        # Instantiate the Bayes learner
        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen
        self.alpha0 = alpha0
        self.mu0 = mu0
        self.tau0 = tau0
        self.tau = tau
        self.observed_reward = 0
        self.__dict__.update(locals())#observed_reward = 0

        self.query_function.setAgent(self)

        self.qVals = {}
        self.qMax = {}

        # Now make the prior beliefs
        self.R_prior = {}
        self.P_prior = {}

        for state in xrange(nState):
            for action in xrange(nAction):
                self.R_prior[state, action] = (self.mu0, self.tau0)
                self.P_prior[state, action] = (
                    self.alpha0 * np.ones(self.nState, dtype=np.float32))

    def update_obs(self, oldState, action, reward, newState, pContinue, h, query=True):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''
        if query:
            mu0, tau0 = self.R_prior[oldState, action]
            tau1 = tau0 + self.tau
            mu1 = (mu0 * tau0 + reward * self.tau) / tau1
            if self.reward_depends_on_action:
                self.R_prior[oldState, action] = (mu1, tau1)
            else:
                for action in range(self.nAction):
                    self.R_prior[oldState, action] = (mu1, tau1)
            self.observed_reward += reward

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

    def egreedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        Q = self.qVals[state, timestep]
        nAction = Q.size
        noise = np.random.rand()

        if noise < epsilon:
            action = np.random.choice(nAction)
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])

        return action

    def pick_action(self, state, timestep):
        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep)
        return action

    def sample_mdp(self):
        '''
        Returns a single sampled MDP from the posterior.

        Args:
            NULL

        Returns:
            R_samp - R_samp[s, a] is the sampled mean reward for (s,a)
            P_samp - P_samp[s, a] is the sampled transition vector for (s,a)
        '''
        R_samp = {}
        P_samp = {}
        for s in xrange(self.nState):
            if not self.reward_depends_on_action:
                mu, tau = self.R_prior[s, 0]
                sample = mu + np.random.normal() * 1./np.sqrt(tau)
            for a in xrange(self.nAction):
                if self.R_true:
                    R_samp[s,a] = self.R_true[s,a][0]
                else:
                    if self.reward_depends_on_action:
                        mu, tau = self.R_prior[s, a]
                        sample = mu + np.random.normal() * 1./np.sqrt(tau)
                    R_samp[s, a] = sample

                if self.P_true is None:
                    P_samp[s, a] = np.random.dirichlet(self.P_prior[s, a])

        if self.P_true is not None:
            P_samp = self.P_true

        return R_samp, P_samp

    # this appears to be the "expected MDP" as I was calling it
    def map_mdp(self):
        '''
        Returns the maximum a posteriori MDP from the posterior.

        Args:
            NULL

        Returns:
            R_hat - R_hat[s, a] is the MAP mean reward for (s,a)
            P_hat - P_hat[s, a] is the MAP transition vector for (s,a)
        '''
        R_hat = {}
        P_hat = {}
        for s in xrange(self.nState):
            for a in xrange(self.nAction):
                R_hat[s, a] = self.R_prior[s, a][0]
                P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])

        return R_hat, P_hat

    
    def compute_qVals(self, R, P):
        '''
        Compute the Q values for a given R, P estimates

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        
        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax


    def compute_qVals_true(self, R, P, R_true, P_true):
        '''
        Evaluate an agent's expected returns when it plans according to R and P
        in an environment defined by R_true, P_true

        Returns:
            The true expected returns of the agent, 
             what it thinks its expected returns are.
        '''
        qVals = {}
        qMax = {} # aka "V"
        qVals_true = {}
        qMax_true = {}

        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        qMax_true[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        
        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)
            qMax_true[j] = np.zeros(self.nState, dtype=np.float32)
             
            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)
                qVals_true[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])
                    qVals_true[s, j][a] = R_true[s, a] + np.dot(P_true[s, a], qMax_true[j + 1])
        
                # agent acts according to what it believes
                a = np.argmax(qVals[s, j])
                # we compute both its estimate of the value of this state/tstep, and the true value
                qMax[j][s] = qVals[s, j][a]
                qMax_true[j][s] = qVals_true[s, j][a]
        
        # M_true, M_prior
        return qMax_true[0][0], qMax[0][0]


    def compute_qVals_opt(self, R, P, R_bonus, P_bonus):
        '''
        Compute the Q values for a given R, P estimates + R/P bonus

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_bonus - R_bonus[s,a] = bonus for rewards
            P_bonus - P_bonus[s,a] = bonus for transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    qVals[s, j][a] = (R[s, a] + R_bonus[s, a]
                                      + np.dot(P[s, a], qMax[j + 1])
                                      + P_bonus[s, a] * i)
                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_EVI(self, R, P, R_slack, P_slack):
        '''
        Compute the Q values for a given R, P by extended value iteration

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_slack - R_slack[s,a] = slack for rewards
            P_slack - P_slack[s,a] = slack for transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
                # Extended value iteration
        qVals = {}
        qMax = {}
        qMax[self.epLen] = np.zeros(self.nState)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    rOpt = R[s, a] + R_slack[s, a]

                    # form pOpt by extended value iteration, pInd sorts the values
                    pInd = np.argsort(qMax[j + 1])
                    pOpt = P[s, a]
                    if pOpt[pInd[self.nState - 1]] + P_slack[s, a] * 0.5 > 1:
                        pOpt = np.zeros(self.nState)
                        pOpt[pInd[self.nState - 1]] = 1
                    else:
                        pOpt[pInd[self.nState - 1]] += P_slack[s, a] * 0.5

                    # Go through all the states and get back to make pOpt a real prob
                    sLoop = 0
                    while np.sum(pOpt) > 1:
                        worst = pInd[sLoop]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1

                    # Do Bellman backups with the optimistic R and P
                    qVals[s, j][a] = rOpt + np.dot(pOpt, qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------
class PSRL(FiniteHorizonTabularAgent):
    '''
    Posterior Sampling for Reinforcement Learning
    '''

    def update_policy(self, h=False):
        '''
        Sample a single MDP from the posterior and solve for optimal Q values.
        Works in place with no arguments.
        '''
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()

        # Solve the MDP via value iteration
        qVals, qMax = self.compute_qVals(R_samp, P_samp)

        # Update the Agent's Q-values
        self.qVals = qVals
        self.qMax = qMax

# TODO: rename! 
class PSRLLimitedQuery(PSRL):
    '''
    Posterior Sampling for Reinforcement Learning
    '''

    def sample_mdp_unclamped(self):
        return FiniteHorizonTabularAgent.sample_mdp(self)

    def sample_mdp(self):
        '''
        Sample MDP but clamp rewards if we're no longer learning them.
        '''
        R_samp, P_samp = FiniteHorizonTabularAgent.sample_mdp(self)
        #import ipdb; ipdb.set_trace()

        def thompson_or_not(sa, r): 
            # FIXME: will_query should use actual episode/tstep
            if self.query_function.will_query(*sa, episode=0, timestep=0):
                return r
            else:
                return self.R_prior[sa][0]

        if self.stop_learning:
            R_samp = { sa : thompson_or_not(sa, r) for sa, r in R_samp.iteritems() }

        return R_samp, P_samp

#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------

class PSRLunif(PSRL):
    '''
    Posterior Sampling for Reinforcement Learning with spread prior
    '''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., **kwargs):
        '''
        Just like PSRL but rescale alpha between successor states

        Args:
            nSamp - int - number of samples to use for optimism
        '''
        newAlpha = alpha0 / nState
        super(PSRLunif, self).__init__(nState, nAction, epLen, alpha0=newAlpha,
                                       mu0=mu0, tau0=tau0, tau=tau)

#-----------------------------------------------------------------------------
# Optimistic PSRL
#-----------------------------------------------------------------------------

class OptimisticPSRL(PSRL):
    '''
    Optimistic Posterior Sampling for Reinforcement Learning
    '''
    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., nSamp=10, **kwargs):
        '''
        Just like PSRL but we take optimistic over multiple samples

        Args:
            nSamp - int - number of samples to use for optimism
        '''
        super(OptimisticPSRL, self).__init__(nState, nAction, epLen,
                                             alpha0, mu0, tau0, tau)
        self.nSamp = nSamp

    def update_policy(self, h=False):
        '''
        Take multiple samples and then take the optimistic envelope.

        Works in place with no arguments.
        '''
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()
        qVals, qMax = self.compute_qVals(R_samp, P_samp)
        self.qVals = qVals
        self.qMax = qMax

        for i in xrange(1, self.nSamp):
            # Do another sample and take optimistic Q-values
            R_samp, P_samp = self.sample_mdp()
            qVals, qMax = self.compute_qVals(R_samp, P_samp)

            for timestep in xrange(self.epLen):
                self.qMax[timestep] = np.maximum(qMax[timestep],
                                                 self.qMax[timestep])
                for state in xrange(self.nState):
                    self.qVals[state, timestep] = np.maximum(qVals[state, timestep],
                                                             self.qVals[state, timestep])

#-----------------------------------------------------------------------------
# Gaussian PSRL
#-----------------------------------------------------------------------------

class GaussianPSRL(FiniteHorizonTabularAgent):
    '''Naive Gaussian approximation to PSRL, similar to tabular RLSVI'''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1.):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super(GaussianPSRL, self).__init__(nState, nAction, epLen, alpha0,
                                    mu0, tau0, tau)
        self.scaling = scaling

    def gen_bonus(self, h=False):
        ''' Generate the Gaussian bonus for Gaussian PSRL '''
        R_bonus = {}
        P_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                R_bonus[s, a] = self.scaling * np.random.normal() * 1. / np.sqrt(R_sum)

                P_sum = self.P_prior[s, a].sum()
                P_bonus[s, a] = self.scaling * np.random.normal() * 1. / np.sqrt(P_sum)

        return R_bonus, P_bonus

    def update_policy(self, h=False):
        '''
        Update Q values via Gaussian PSRL.
        This performs value iteration but with additive Gaussian noise.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Purely Gaussian perturbations
        R_bonus, P_bonus = self.gen_bonus(h)

        # Form approximate Q-value estimates
        qVals, qMax = self.compute_qVals_opt(R_hat, P_hat, R_bonus, P_bonus)

        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# UCBVI
#-----------------------------------------------------------------------------

class UCBVI(GaussianPSRL):
    '''Upper confidence bounds value iteration... similar to Gaussian PSRL'''

    def gen_bonus(self, h=1):
        ''' Generate the sqrt(n) bonus for UCBVI '''
        R_bonus = {}
        P_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                R_bonus[s, a] = self.scaling * np.sqrt(2. * np.log(2 + h) / R_sum)

                P_sum = self.P_prior[s, a].sum()
                P_bonus[s, a] = self.scaling * np.sqrt(2. * np.log(2 + h) / P_sum)

        return R_bonus, P_bonus

#-----------------------------------------------------------------------------
# BEB
#-----------------------------------------------------------------------------

class BEB(GaussianPSRL):
    '''BayesExploreBonus BEB algorithm'''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1.):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super(BEB, self).__init__(nState, nAction, epLen,
                                                alpha0, mu0, tau0, tau)
        self.beta = 2 * self.epLen * self.epLen * scaling

    def gen_bonus(self, h=False):
        ''' Generate the 1/n bonus for BEB'''
        R_bonus = {}
        P_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                R_bonus[s, a] = 1. / (R_sum + 1)

                P_sum = self.P_prior[s, a].sum()
                P_bonus[s, a] = self.beta * self.epLen / (1 + P_sum)

        return R_bonus, P_bonus

#-----------------------------------------------------------------------------
# BOLT
#-----------------------------------------------------------------------------

class BOLT(FiniteHorizonTabularAgent):
    '''Bayes Optimistic Local Transitions (BOLT)'''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1.):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super(BOLT, self).__init__(nState, nAction, epLen,
                                    alpha0, mu0, tau0, tau)
        self.eta = self.epLen * scaling

    def get_slack(self, time):
        '''
        Returns the slackness parameters for BOLT.
        These are based upon eta imagined optimistic observations

        Args:
            time - int - grows the confidence sets

        Returns:
            R_slack - R_slack[s, a] is the confidence width for BOLT reward
            P_slack - P_slack[s, a] is the confidence width for BOLT transition
        '''
        R_slack = {}
        P_slack = {}

        for s in xrange(self.nState):
            for a in xrange(self.nAction):
                R_slack[s, a] = self.eta / (self.R_prior[s, a][1] + self.eta)
                P_slack[s, a] = 2 * self.eta / (self.P_prior[s, a].sum() + self.eta)
        return R_slack, P_slack

    def update_policy(self, h=False):
        '''
        Compute BOLT Q-values via extended value iteration.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slack parameters
        R_slack, P_slack = self.get_slack(h)

        # Perform extended value iteration
        qVals, qMax = self.compute_qVals_EVI(R_hat, P_hat, R_slack, P_slack)

        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# UCRL2
#-----------------------------------------------------------------------------

class UCRL2(FiniteHorizonTabularAgent):
    '''Classic benchmark optimistic algorithm'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        super(UCRL2, self).__init__(nState, nAction, epLen,
                                    alpha0=1e-5, tau0=0.0001)
        self.delta = delta
        self.scaling = scaling


    def get_slack(self, time):
        '''
        Returns the slackness parameters for UCRL2

        Args:
            time - int - grows the confidence sets

        Returns:
            R_slack - R_slack[s, a] is the confidence width for UCRL2 reward
            P_slack - P_slack[s, a] is the confidence width for UCRL2 transition
        '''
        R_slack = {}
        P_slack = {}
        delta = self.delta
        scaling = self.scaling
        for s in xrange(self.nState):
            for a in xrange(self.nAction):
                nObsR = max(self.R_prior[s, a][1] - self.tau0, 1.)
                R_slack[s, a] = scaling * np.sqrt((4 * np.log(2 * self.nState * self.nAction * (time + 1) / delta)) / float(nObsR))

                nObsP = max(self.P_prior[s, a].sum() - self.alpha0, 1.)
                P_slack[s, a] = scaling * np.sqrt((4 * self.nState * np.log(2 * self.nState * self.nAction * (time + 1) / delta)) / float(nObsP))
        return R_slack, P_slack

    def update_policy(self, time=100):
        '''
        Compute UCRL2 Q-values via extended value iteration.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slack parameters
        R_slack, P_slack = self.get_slack(time)

        # Perform extended value iteration
        qVals, qMax = self.compute_qVals_EVI(R_hat, P_hat, R_slack, P_slack)

        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# UCFH
#-----------------------------------------------------------------------------

class UCFH(UCRL2):
    '''Dann+Brunskill modificaitons to UCRL2 for finite domains'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., epsilon=0.1, **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        super(UCFH, self).__init__(nState, nAction, epLen,
                                   alpha0=1e-9, tau0=0.0001)
        self.epsilon = epsilon
        self.delta = delta
        self.scaling = scaling
        self.epsilon = epsilon
        wMin = epsilon / (4 * nState * epLen)
        uMax = nState * nAction * np.log(nState * epLen / wMin) / np.log(2)
        self.delta1 = delta / (2 * uMax * nState)

    def compute_confidence(self, pHat, n):
        '''
        Compute the confidence sets for a give p component.
        Dann + Brunskill style

        Args:
            pHat - estimated transition probaility component
            n - number of observations
            delta - confidence paramters

        Returns:
            valid_p
        '''
        delta1 = self.delta1
        scaling = self.scaling
        target_sd = np.sqrt(pHat * (1 - pHat))
        K_1 = scaling * np.sqrt(2 * np.log(6 / delta1) / float(max(n - 1, 1)))
        K_2 = scaling * target_sd * K_1 + 7 / (3 * float(max(n - 1, 1))) * np.log(6 / delta1)

        sd_min = target_sd - K_1
        C_1 = (target_sd - K_1) * (target_sd - K_1)
        varLower, varUpper = (0, 1)

        # Only look after one side of variance inequality since Dann+Brunskill
        # algorithm ignores the other side anyway
        if sd_min > 1e-5 and C_1 > 0.2499:
            varLower = 0.5 * (1 - np.sqrt(1 - 4 * C_1))
            varUpper = 0.5 * (1 + np.sqrt(1 - 4 * C_1))

        # Empirical mean constrains
        mean_min = pHat - K_2
        mean_max = pHat + K_2

        # Checking the type of contstraint
        if pHat < varLower or pHat > varUpper:
            varLower, varUpper = (0, 1)

        # Don't worry about non-convex interval, since it is not used in paper
        interval = [np.max([0, varLower, mean_min]),
                    np.min([1, varUpper, mean_max])]
        return interval


    def update_policy(self, time=100):
        '''
        Updates the policy with UCFH extended value iteration
        '''
        # Extended value iteration
        qVals = {}
        qMax = {}
        qMax[self.epLen] = np.zeros(self.nState)

        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slackness for rewards UCRL2 style
        R_slack = {}
        delta = self.delta
        delta1 = self.delta1
        scaling = self.scaling
        for s in xrange(self.nState):
            for a in xrange(self.nAction):
                nObsR = max(self.R_prior[s, a][1] - self.tau0, 1.)
                R_slack[s, a] = scaling * np.sqrt((4 * np.log(2 * self.nState * self.nAction * (time + 1) / delta)) / nObsR)

        P_range = {}
        # Extended value iteration as per Dann+Brunskill
        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    nObsP = max(self.P_prior[s, a].sum() - self.alpha0, 1.)
                    rOpt = R_hat[s, a] + R_slack[s, a]
                    pOpt = np.zeros(self.nState)

                    # pInd sorts the next-step values in *increasing* order
                    pInd = np.argsort(qMax[j + 1])

                    for sPrime in range(self.nState):
                        P_range[s, a, sPrime] = self.compute_confidence(P_hat[s,a][sPrime], nObsP)
                        pOpt[sPrime] = P_range[s, a, sPrime][0]

                    pSlack = 1 - pOpt.sum()

                    if pSlack < 0:
                        print 'ERROR we have a problem'

                    for sPrime in range(self.nState):
                        # Reverse the ordering
                        newState = pInd[self.nState - sPrime - 1]
                        newSlack = min([pSlack, P_range[s, a, newState][1] - pOpt[newState]])
                        pOpt[newState] += newSlack
                        pSlack -= newSlack
                        if pSlack < 0.001:
                            break
                    qVals[s, j][a] = rOpt + np.dot(pOpt, qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])
        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# Epsilon-Greedy
#-----------------------------------------------------------------------------

class EpsilonGreedy(FiniteHorizonTabularAgent):
    '''Epsilon greedy agent'''

    def __init__(self, nState, nAction, epLen, epsilon=0.1, **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            epsilon - double - probability of random action
        '''
        super(EpsilonGreedy, self).__init__(nState, nAction, epLen,
                                            alpha0=0.0001, tau0=0.0001)
        self.epsilon = epsilon

    def update_policy(self, time=False):
        '''
        Compute UCRL Q-values via extended value iteration.

        Args:
            time - int - grows the confidence sets
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Solve the MDP via value iteration
        qVals, qMax = self.compute_qVals(R_hat, P_hat)

        # Update the Agent's Q-values
        self.qVals = qVals
        self.qMax = qMax

    def pick_action(self, state, timestep):
        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep, self.epsilon)
        return action

#------------------------------------------------------------------------------
# Q-learning with optimistic initialization
#------------------------------------------------------------------------------

class OptimisticQLearner(FiniteHorizonTabularAgent):
    ''' Naive Q-learning online with optimistic initialization'''

    def __init__(self, nState, nAction, epLen,
                 learnRate=0.01, epsilon=0.05, qInit=False):
        # Fill in the variables
        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen
        self.learnRate = learnRate
        self.epsilon = epsilon

        if not qInit:
            self.qInit = self.epLen
        else:
            self.qInit = qInit

        qVals = {}
        for state in xrange(self.nState):
            for timestep in xrange(self.epLen + 1):
                qVals[state, timestep] = np.ones(nAction, dtype=np.float32) * qInit
        self.qVals = qVals

    def update_obs(self, oldState, action, reward, newState, pContinue, timestep):
        '''
        Update the Q values by simple online Q-learning

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            timestep - int - time within episode

        Returns:
            NULL - updates in place
        '''
        oldQ = self.qVals[oldState, timestep][action]
        newQ = reward + pContinue * self.qVals[newState, timestep + 1].max()
        self.qVals[oldState, timestep][action] = oldQ + self.learnRate * (newQ - oldQ)

    def pick_action(self, state, timestep):
        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep, self.epsilon)
        return action


