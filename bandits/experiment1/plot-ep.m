mus = [0.6 0.5 0.4 0.4];
n = 10000; % horizon
k = length(mus);
cost = 2;

load('ep_policy1.mat');
ep_policy1 = ep;
load('ep_epsquery.mat');
ep_epsquery = ep;
load('ep_fixedquery.mat');
ep_fixedquery = ep;
load('ep_expquery.mat');
ep_expquery = ep;
load('ep_ocucb.mat');
ep_ocucb = ep;


loglog(10:10:n, mean(ep_policy1)(10:10:n), '-',
       10:10:n, mean(ep_epsquery)(10:10:n), '-',
       10:10:n, mean(ep_fixedquery)(10:10:n), '-',
       10:10:n, mean(ep_expquery)(10:10:n), '-',
       10:n,    mean(ep_ocucb)(10:n), '-',
       10:n, sqrt(2 / pi) ./ sqrt(10:n), '--');
legend(sprintf('Policy1, N=%d', size(ep_policy1)(1)),
       sprintf('Eps-query+OCUCB, N=%d', size(ep_epsquery)(1)),
       sprintf('sqrt(n/cost)-query+OCUCB, N=%d', size(ep_fixedquery)(1)),
       sprintf('Exp-query+OCUCB, N=%d', size(ep_expquery)(1)),
       sprintf('OCUCB (cost=0), N=%d', size(ep_ocucb)(1)),
       'sqrt(2/(pi*t))',
       'location', 'southwest');
xlabel('time step');
ylabel('average exploration potential');
title(sprintf('Active Bernoulli bandit with arms [%s], cost = %f', num2str(mus), cost));
