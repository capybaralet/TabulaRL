% algorithms for n-armed active bandits

1;

function j = EpsGreedyPolicy(T, s, n, t, d = 0.1, c = 0.05)
    k = length(T); % number of arms
    eps_t = c*k / d^2 / t; # from the theory, d is the minimum gap in the arms
%    eps_t = 1/t;
    if rand() > eps_t
        # greedy policy
        mu_hats = (s + 1) ./ (T + 2);
        [_ j] = max(mu_hats);
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

function j = Arm1Policy(T, s, n, t)
    % policy that always pulls arm 1
    j = 1;
end

function [j query] = activeBanditPolicy1(T, s, n, t, cost, c = 1/2)
	% shitty heuristic
	k = length(T); % number of arms
	if any(T == 0)
		j = find(T == 0, 1);
		query = true;
	else
		ucb = s ./ T .* (1 + (s + 1) ./ (T + 2) - s ./ T) - cost * k / (n - t + 1);
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

function [j query] = FixedQueryPolicy(T, s, n, t, cost, alg=@OCUCBPolicy)
    % use standard and query the first sqrt(n/cost) steps
    if t <= sqrt(n/cost)
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
        query = T(j) <= max(T(idx));
    else
        query = true;
    end
end

function [j query] = activeBanditPolicy2(T, s, n, t, cost)
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


