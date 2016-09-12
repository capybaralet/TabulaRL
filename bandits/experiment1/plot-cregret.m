mus = [0.6 0.5 0.4 0.4];
n = 10000; % horizon
k = length(mus);
cost = 2;

load('regret_policy1.mat');
regret_policy1 = regret;
load('regret_epsquery.mat');
regret_epsquery = regret;
load('regret_fixedquery.mat');
regret_fixedquery = regret;
load('regret_expquery.mat');
regret_expquery = regret;
load('regret_ocucb.mat');
regret_ocucb = regret;


plot(0:n, [0 mean(regret_policy1)], '-',
     0:n, [0 mean(regret_epsquery)], '-',
     0:n, [0 mean(regret_fixedquery)], '-',
     0:n, [0 mean(regret_expquery)], '-',
     0:n, [0 mean(regret_ocucb)], '-');
%ylim([0 200]);
legend(sprintf('Policy1, N=%d', size(regret_policy1)(1)),
       sprintf('Eps-query+OCUCB, N=%d', size(regret_epsquery)(1)),
       sprintf('sqrt(n/cost)-query+OCUCB, N=%d', size(regret_fixedquery)(1)),
       sprintf('Exp-query+OCUCB, N=%d', size(regret_expquery)(1)),
       sprintf('OCUCB (cost=0), N=%d', size(regret_ocucb)(1)),
       'location', 'southeast');
xlabel('time step');
ylabel('average cumulative regret');
title(sprintf('Active Bernoulli bandit with arms [%s], cost = %f', num2str(mus), cost));

