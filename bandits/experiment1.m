% Experiment 1

source('bandits.m');

global mus = [0.6 0.5 0.4 0.4]; % means of the arms
global n = 10000; % horizon
global cost = 2; % query cost
global k = length(mus); % number of arms
global policy = @parameterizedRegretQuery;
global save_ep = false; % whether to compute the exploration potential (very slow!)
N = 100; % number of repetitions

assert(nargin >= 1);
savefile = argv(){1};

function [cregret, ep] = playBernoulli(mu, n, policy, cost)
    % play a game of bernoulli arms
    % n = horizon
    % mu = list of Bernoulli parameters
    k = length(mu); % number of arms
    T = zeros(1, k); % number of times pulled each arm
    s = zeros(1, k); % number of successes on each arm
    regret = 0; % cumulative undiscounted regret
    cregret = zeros(1, n);
    ep = zeros(1, n);
    mu_best = max(mu);
    printf('t =  ');
    bsp = "\b";
    for t = 1:n
        if length(num2str(t - 1)) > length(bsp)
            bsp = strcat(bsp, "\b");
        end
        printf("%s%d", bsp, t);
        fflush(stdout);
        [j query] = policy(T, s, n, t, cost);

        % Pull arm j
        r = mu(j) > rand(); % reward = 0 or 1
        if query
            T(j) += 1;
            s(j) += r;
            regret += cost;
            regret += mu_best - mu(j);
            cregret(t) = regret;
        else
            printf(" commited to arm %d", j);
            for i = t:n
                regret += mu_best - mu(j);
                cregret(i) = regret;
            end
            break;
        end
        global save_ep;
        if save_ep && mod(t, 10) == 0
            ep(t) = EP(T, s);
        end
    end
    printf("\n");
end

function [regret, ep] = runExperiment1(mus, N, n, policy, cost)
    % mus: list of the arm's means
    % N: number of data points for each epsilon
    % n: horizon
    % policy: the bandit policy
    regret = zeros(N, n);
    ep = zeros(N, n);
    for i = 1:N
        [regret(i, :) ep(i, :)] = playBernoulli(mus, n, policy, cost);
    end
end

display(ctime(time()));
fflush(stdout);
[regret, ep] = runExperiment1(mus, N, n, policy, cost);
save("-z", sprintf('experiment1/regret_%s.mat', savefile), 'regret');
if save_ep
    save("-z", sprintf('experiment1/ep_%s.mat', savefile), 'ep');
end
printf("Finished.\n");
display(ctime(time()));
