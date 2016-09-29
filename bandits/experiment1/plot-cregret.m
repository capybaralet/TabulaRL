mus = [0.6 0.5 0.4 0.4];
n = 10000; % horizon
k = length(mus);
cost = 2;

load('regret_policy1.mat');
regret_policy1 = regret;
load('regret_policy2.mat');
regret_policy2 = regret;
load('regret_epsquery.mat');
regret_epsquery = regret;
load('regret_fixedquery.mat');
regret_fixedquery = regret;
load('regret_23fixedquery.mat');
regret_23fixedquery = regret;
load('regret_expquery.mat');
regret_expquery = regret;
load('regret_ocucb.mat');
regret_ocucb = regret;
load('regret_dmedstop.mat');
regret_dmed = regret;
load('regret_paramRegret35.mat');
regret_paramRegret = regret;


plot(0:n, [0 mean(regret_policy1)], '-',
     0:n, [0 mean(regret_policy2)], '-',
     0:n, [0 mean(regret_epsquery)], '-',
     0:n, [0 mean(regret_fixedquery)], '-',
%     0:n, [0 mean(regret_23fixedquery)], '-',
     0:n, [0 mean(regret_expquery)], '-',
     0:n, [0 mean(regret_ocucb)], '-',
%     0:n, [0 mean(regret_dmed)], '-',
     0:n, [0 mean(regret_paramRegret)], '-');
%ylim([0 200]);
legend(sprintf('Policy1, N=%d', size(regret_policy1)(1)),
       sprintf('Policy2, N=%d', size(regret_policy2)(1)),
       sprintf('Eps-query+OCUCB, N=%d', size(regret_epsquery)(1)),
       sprintf('sqrt(n/cost)-query+OCUCB, N=%d', size(regret_fixedquery)(1)),
%       sprintf('(n/cost)^(2/3)-query+OCUCB, N=%d', size(regret_23fixedquery)(1)),
       sprintf('Exp-query+OCUCB, N=%d', size(regret_expquery)(1)),
       sprintf('OCUCB (cost=0), N=%d', size(regret_ocucb)(1)),
%       sprintf('DMED with cost, N=%d', size(regret_dmed)(1)),
       sprintf('parameterized Regret, alpha=0.35, N=%d', size(regret_paramRegret)(1)),
       'location', 'north');
xlabel('time step');
ylabel('average cumulative regret');
title(sprintf('Active Bernoulli bandit with arms [%s], cost = %f', num2str(mus), cost));
