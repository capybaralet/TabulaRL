% algorithms for n-armed active bandits

1;

function j = bestArm(T, s)
    % Return the best arm
    mu_hats = (s + 1) ./ (T + 2);
	[_ j] = max(mu_hats);
end

function j = EpsGreedyPolicy(T, s, n, t, d = 0.1, c = 0.05)
    k = length(T); % number of arms
    eps_t = c*k / d^2 / t; # from the theory, d is the minimum gap in the arms
%    eps_t = 1/t;
    if rand() > eps_t
        # greedy policy
        j = bestArm(T, s);
    else
        # pick an action uniformly at random
        j = randi(k);
    end
end

function j = ThompsonPolicy(T, s, n, t)
    % policy based on Thompson sampling
    % T: number of times pulled each arm
    % s: number of successes on each arm
    % n: horizon
    % t: current time step
    k = length(T); % number of arms
    assert(k == length(s));
    mu_hats = betarnd(s + 1, T - s + 1);
    [_ j] = max(mu_hats);
end

function j = UCBPolicy(T, s, n, t, c = 1/2)
    % policy based on the UCB algorithm
    % T: number of times pulled each arm
    % s: number of successes on each arm
    % n: horizon
    % t: current time step
    % c: parameter to UCB
    if any(T == 0)
        j = find(T == 0, 1);
    else
        ucb =  s ./ T + sqrt(c * log(t) ./ T);
        m = max(ucb);
        idx = find(ucb == m);
        j = idx(ceil(length(idx) * rand()));
    end
end

function j = UCBaltPolicy(T, s, n, t, c = 1/2)
    % alternative policy based on the UCB algorithm with confidence n / T_i
    % T: number of times pulled each arm
    % s: number of successes on each arm
    % n: horizon
    % t: current time step
    % c: parameter to UCB
    if any(T == 0)
        j = find(T == 0, 1);
    else
        ucb =  s ./ T + sqrt(c * log(n ./ T) ./ T);
        m = max(ucb);
        idx = find(ucb == m);
        j = idx(ceil(length(idx) * rand()));
    end
end

function j = OCUCBPolicy(T, s, n, t, c = 1/2)
    % policy based on the optimally confident UCB algorithm by Tor Lattimore
    % T: number of times pulled each arm
    % s: number of successes on each arm
    % n: horizon
    % t: current time step
    % c: parameter to UCB
    if any(T == 0)
        j = find(T == 0, 1);
    else
        ucb =  s ./ T + sqrt(c * log(n / t) ./ T);
        m = max(ucb);
        idx = find(ucb == m);
        j = idx(1 + floor(length(idx) * rand()));
    end
end

function j = BayesUCBPolicy(T, s, n, t)
    % Bayes-UCB policy with quartile 1/t
    % T: number of times pulled each arm
    % s: number of successes on each arm
    % n: horizon
    % t: current time step
    if any(T == 0)
        j = find(T == 0, 1);
    else
        quantiles = betaincinv(1 - 1/t, s + 1, T - s + 1);
        m = max(quantiles);
        idx = find(quantiles == m);
        j = idx(ceil(length(idx) * rand()));
    end
end

function [j query] = Arm1Policy(T, s, n, t)
    % policy that always pulls arm 1
    j = 1;
    query = false;
end

function [j query] = activeBanditPolicy1(T, s, n, t, cost, c = 1/2)
	% shitty heuristic
	k = length(T); % number of arms
	if any(T == 0)
		j = find(T == 0, 1);
		query = true;
	else
		ucb = s ./ T + sqrt(c * log(n / t) ./ T) - cost * T / (n - t + 1);
		m = max(ucb);
		mu_hats = (s + 1) ./ (T + 2);
		idx = find(ucb == m);
		if m <= max(mu_hats)
			[_, j] = max(mu_hats);
			query = false;
		else
			j = idx(1 + floor(length(idx) * rand()));
			query = true;
		end
	end
end

function [j query] = activeBanditPolicy2(T, s, n, t, cost, c = 1/2)
	% shitty heuristic
	k = length(T); % number of arms
	if any(T == 0)
		j = find(T == 0, 1);
		query = true;
	else
		ucb = s ./ T + sqrt(c * log(n / t) ./ T) - cost * sqrt(T / n);
		m = max(ucb);
		mu_hats = (s + 1) ./ (T + 2);
		idx = find(ucb == m);
		j = idx(1 + floor(length(idx) * rand()));
		idx2 = 1:k;
		idx2([j]) = [];
		if m > max(mu_hats(idx2)) || T(j) < max(T(idx2))
			query = true;
		else
			[_, j] = max(mu_hats);
			query = false;
		end
	end
end


function y = f(theta, a, b, j, arm, tol)
    % integrant for E [ theta_j - theta_arm | j is best arm ]
    assert(j != arm);
    k = length(a);
    y = theta .* betacdf(theta, a(arm), b(arm));
    y -= quadgk(@(x) x .* betapdf(x, a(arm), b(arm)), 0, theta, tol);
    y .*= betapdf(theta, a(j), b(j));
    for s = 1:k
        if (s != arm && s != j)
            y .*= betacdf(theta, a(s), b(s));
        end
    end
end

function x = expectedRegret(T, s, n, t, arm, tol=1e-3)
    % Bayes-expected regret when committing to the arm "arm"
    k = length(T);
    x = 0;
    for j = 1:k
        if j != arm
            x += quad(@(x) f(x, s + 1, T - s + 1, j, arm, tol), 0, 1, tol);
        end
    end
    x *= n - t;
end

function y = g(x, a, b, arm)
    k = length(a);
    y = betapdf(x, a(arm), b(arm));
    for s = 1:k
        if (s != arm)
            y .*= betacdf(x, a(s), b(s));
        end
    end
end

function x = probBestArm(T, s, arm, tol=1e-3)
    % posterior probability that "arm" is the best arm
    k = length(T);
    x = quad(@(x) g(x, s + 1, T - s + 1, arm), 0, 1, tol);
end

function x = minExpectedRegret(T, s, n, t, tol=1e-3)
    % Bayes-expected regret when committing to the best arm
    x = expectedRegret(T, s, n, t, bestArm(T, s), tol);
end

function [j query] = activeBanditPolicy3(T, s, n, t, cost, c = 1/2)
	% Bayesian bliss
    % FIXME: problem that the lookahead is only 1 step
	k = length(T); % number of arms

	% compute the Bayes-expected regret of committing to an arm and compare it to querying
	query = false;
    j = bestArm(T, s);
    mu_hats = (s + 1) ./ (T + 2);
    m = expectedRegret(T, s, n, t, j);
    for i = 1:k
        e_i = zeros(1, k);
        e_i(i) = 1;
        T_next = T + e_i;
        if i == j
            rg_i = minExpectedRegret(T_next, s + e_i, n, t) * mu_hats(i);
        else
            rg_i = minExpectedRegret(T_next, s, n, t) * (1 - mu_hats(i));
        end
        if rg_i + cost < m
            m = rg_i;
            j = i;
            query = true;
        end
    end
end

function [j query] = DMEDPolicy(T, s, n, t, cost)
    % Honda, Junya, and Akimichi Takemura.
    % An Asymptotically Optimal Bandit Algorithm for Bounded Support Models.
    % COLT 2010.
    % here we use (theta - theta^*) instead of KL(B(theta), B(theta^*))
    k = length(T);
    mu_hats = (s + 1) ./ (T + 2);
    m = max(mu_hats);
    Jprime = (T .* (m - mu_hats) <= log(n) - log(T));
    arms = (1:k)(Jprime);
    %j = arms(mod(t, length(arms)) + 1);
    j = arms(1 + floor(length(arms) * rand()));
    query = length(arms) > 1;
end

function [j query] = DMEDStopPolicy(T, s, n, t, cost)
    % variant on DMEDPolicy that stops when
    % 2*(cost * min_gap_steps^2) + regret_after > minExpectedRegret(T, s, n, t)
    k = length(T);
    mu_hats = (s + 1) ./ (T + 2);
    [m best_arm] = max(mu_hats);
    [min_gap_steps i] = min(((T + 1) .* (m - mu_hats))([(1:(best_arm-1)) ((best_arm+1):k)]));
    min_gap_steps = ceil(min_gap_steps + 0.01);
    if t + min_gap_steps^2 >= n
       % instant commitment because the time frame is too long
       j = best_arm;
       query = false;
    end
    e_i = zeros(1, k);
    e_i(i) = 1;
    need_successes = ceil(mu_hats(i) * min_gap_steps^2 + min_gap_steps);
    if need_successes > min_gap_steps^2
        need_successes = min_gap_steps^2;
    end
    regret_after = minExpectedRegret(T + min_gap_steps^2 * e_i, s + need_successes * e_i, n, t + min_gap_steps^2);
    mu_hats
    [t min_gap_steps (2*(cost * min_gap_steps^2) + regret_after) (minExpectedRegret(T, s, n, t) + 1e-6)]
    query = (2*(cost * min_gap_steps^2) + regret_after < minExpectedRegret(T, s, n, t) + 1e-6);
    if query
        Jprime = (T .* (m - mu_hats) <= log(n) - log(T));
        arms = (1:k)(Jprime);
        %j = arms(mod(t, length(arms)) + 1);
        j = arms(1 + floor(length(arms) * rand()));
    else
        j = best_arm;
    end
end

function n = nonlinearSequentialElim(T, s, n, t, cost, p = 1.5)
    % From https://arxiv.org/abs/1609.02606
    assert(p > 0);
    k = length(T);
    budget = n^(2/3);
    C_p = sum([2 2:k] .^ (-p));
    n = ceil((budget - k) / C_p * (k - (0:(k-2))) .^ (-p));
end

function [j query] = FixedQueryPolicy(T, s, n, t, cost, alg=@OCUCBPolicy)
    % use standard and query the first sqrt(n/cost) steps
    if t <= (n/cost)^(2/3)
        query = true;
        j = alg(T, s, n, t);
    else
        query = false;
        [_, j] = max((s + 1) ./ (T + 2));
    end
end

function [j query] = EpsQueryPolicy(T, s, n, t, cost, alg=@OCUCBPolicy)
    % use OCUCB and query the first sqrt(n/cost) steps
    if rand() < 1/t
        query = true;
        j = alg(T, s, n, t);
    else
        query = false;
        [_, j] = max((s + 1) ./ (T + 2));
    end
end

function [j query] = ExpQueryPolicy(T, s, n, t, cost, alg=@OCUCBPolicy)
    % query whenever doing an exploration action
    mu_hats = (s + 1) ./ (T + 2);
	k = length(T); % number of arms
    j = alg(T, s, n, t);
    if mu_hats(j) == max(mu_hats)
        idx = 1:k;
        idx([j]) = [];
        query = T(j) < max(T(idx));
    else
        query = true;
    end
end

function s = querySteps(T, s)
    % the number of steps you expect to need to
    % bring the two arms with the highest means together
    % (crude heuristic)
    k = length(T);
    mu_hats = (s + 1) ./ (T + 2);
    [m j] = max(mu_hats);
    min_gap_steps = min(((T + 1) .* (m - mu_hats))([(1:(j-1)) ((j+1):k)]));
    s = 2*ceil(min_gap_steps + 0.01)^2;
end

function s = querySteps2(T, s)
    % the number of steps you expect to need to
    % bring the two arms with the highest means together
    % (slightly less crude heuristic)
    k = length(T);
    mu_hats = (s + 1) ./ (T + 2);
    [m j] = max(mu_hats);
    [_ i] = min((m - mu_hats)([(1:(j-1)) ((j+1):k)]));
    i += (i >= j);
    assert(i != j);
    z = 2*(min([T(i) T(j)]) + 2); % upper bound on the number of steps
    % solve the nonlinear programming problem
    % min x1^2 + x2^2
    % s.t. 0 <= x(1) <= z
    %      0 <= x(2) <= z
    %      (s(i) + x(1) + 2*x(1)^2*mu_hats(i) + 1) / (T(i) + 2*x(1)^2 + 2)
    %      >= (s(j) - x(2) + 2*x(2)^2*mu_hats(j) + 1) / (T(j) + 2*x(2)^2 + 2)
    x = sqp([0; 0], {@(x) x(1)^2 + x(2)^2, @(x) 2*x, @(x) 2*eye(2)}, [], @(x) (s(i) + x(1) + 2*x(1)^2*mu_hats(i) + 1) * (T(j) + 2*x(2)^2 + 2) - (s(j) - x(2) + 2*x(2)^2*mu_hats(j) + 1) * (T(i) + 2*x(1)^2 + 2), [0; 0], [z; z]);
    if x == [0; 0]
        x = [1; 0];
    end
    s = 2*ceil(x(1))^2 + 2*ceil(x(2))^2;
end

function s = querySteps3(T, s)
    % the number of steps you expect to need to
    % bring the two arms with the highest means together
    % explicit calculation
    % note: this code is horribly inefficient, but that should't matter ^_^
    k = length(T);
    mu_hats = (s + 1) ./ (T + 2)
    [m j] = max(mu_hats);
    [_ i] = min((m - mu_hats)([(1:(j-1)) ((j+1):k)]));
    i += (i >= j);
    assert(i != j);
    z = sqrt(2*(min([T(i) T(j)]) + 2)); % upper bound on the number of steps
    xi = repmat(0:z, z+1, 1);
    xi2 = xi .^ 2;
    xj = repmat(0:z, z+1, 1)';
    xj2 = xj .^ 2;
    % compute the full matrix of possibilities
    M = (s(i) + xi + 2*xi2*mu_hats(i) + 1) ./ (T(i) + 2*xi2 + 2) - (s(j) - xj + 2*xj2*mu_hats(j) + 1) ./ (T(j) + 2*xj2 + 2)
    for x = 1:z+1
        if any(M(x, 1:x) > 0)
            y = find(M(x, 1:x) > 0)(1);
            s = 2*(x - 1)^2 + 2*(y - 1)^2;
            return;
        end
    end
    s = inf;
end

function [j query] = parameterizedRegretQuery(T, s, n, t, cost, alpha = 0.45)
    % variant on DMEDPolicy that stops when
    % cost to move posterior < alpha * expected regret
    % with parameter alpha \in (0, 1)
    k = length(T);
    mu_hats = (s + 1) ./ (T + 2);
    [m best_arm] = max(mu_hats);
    query_steps = querySteps3(T, s);
    if t + query_steps >= n
       % instant commitment because the time frame is too long
       j = best_arm;
       query = false;
    end
    %[t query_steps minExpectedRegret(T, s, n, t)]
    query = cost * query_steps < alpha * minExpectedRegret(T, s, n, t);
    if query
        % Run DMED
        Jprime = (T .* (m - mu_hats) <= log(n) - log(T));
        arms = (1:k)(Jprime);
        %j = arms(mod(t, length(arms)) + 1);
        j = arms(1 + floor(length(arms) * rand()));
    else
        j = best_arm;
    end
end

function [j query] = activeBanditPolicyX(T, s, n, t, cost)
	% "pretend to commit for k steps"
	mu_hats = (s + 1) ./ (T + 2);
	for i = 1:n-t
		q = 0;
		for a = 0:i
#			q += betapdf(mu_hats, a, i-a)*();
# TODO...
		end
	end
end

function m = minNextEP(T, s)
    k = length(T); % number of arms
    mu_hats = (s + 1) ./ (T + 2);
    m = inf;
    for i = 1:k
        e_i = zeros(1, k);
        e_i(i) = 1;
        m_i = EP(T + e_i, s + e_i) * mu_hats(i) + EP(T + e_i, s) * (1 - mu_hats(i));
        if m_i < m
            m = m_i;
        end
    end
end

function cregret = playBernoulli(mu, n, policy, cost)
    % play a game of bernoulli arms
    % n = horizon
    % mu = list of Bernoulli parameters
    k = length(mu); % number of arms
    T = zeros(1, k); % number of times pulled each arm
    s = zeros(1, k); % number of successes on each arm
    regret = 0; % cumulative undiscounted regret
    cregret = zeros(1, n);
    mu_best = max(mu);
    query = true;
    for t = 1:n
        old_query = query;
        mu_hats = (s + 1) ./ (T + 2);
        [m j] = max(mu_hats);
        if mod(t, 5) == 0
#		    [m best_arm] = max(mu_hats);
#		    mu_hats
#			[min_gap_steps i] = min((T .* (m - mu_hats))([(1:(j-1)) ((j+1):k)]));
#			min_gap_steps = ceil(min_gap_steps + 0.01)
#			e_i = zeros(1, k);
#			e_i(i) = 1;
#			need_successes = ceil(mu_hats(i) * min_gap_steps^2 + min_gap_steps);
#			if need_successes > min_gap_steps^2
#				need_successes = min_gap_steps^2;
#			end
#			regret_after = minExpectedRegret(T + min_gap_steps^2 * e_i, s + need_successes * e_i, n, t + min_gap_steps^2);
#            display([t (2*(cost * min_gap_steps^2) + regret_after) minExpectedRegret(T, s, n, t)]);
            fflush(stdout);
        end
        if t == 70
            [T s]
            break;
        end
#        if mod(t, 10) == 0
#            display([t (EP(T, s) - minNextEP(T, s))*(n - t)]);
#            fflush(stdout);
#        end
        [j query] = policy(T, s, n, t, cost);

        % Pull arm j
        r = mu(j) > rand(); % reward = 0 or 1
        if query
            T(j) += 1;
            s(j) += r;
            regret += cost;
        elseif old_query
            t
        end
        regret += mu_best - mu(j);
        cregret(t) = regret;
    end
end
