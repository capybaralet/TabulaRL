function x = EP(T, s, tol = 1e-12)
    % Computes exploration potential
    % T: number of times pulled each arm
    % s: number of successes on each arm
    % tol: accuracy of the computation
    % 1e-5 is enough for horizon 100
    % 1e-9 is enough for horizon 10000
    % 1e-13 should be enough for horizon 1e6
    k = length(T); % number of arms
    function y = f(theta, mu_hats, a, b, j)
        % function to integrate
        y = abs(theta - mu_hats(j)) .* betapdf(theta, a(j), b(j));
        for i = 1:k
            if (i != j)
                y .*= betacdf(theta, a(i), b(i));
            end
        end
    end
    x = 0;
    mu_hats = (s + 1) ./ (T + 2);
    for j = 1:k
        x += quadgk(@(x) f(x, mu_hats, s + 1, T - s + 1, j), 0, 1, tol);
    end
end
