% solve I(Y;Z) = B to find P(Z|Y)
% sum(sum(A .* log2(A))) - B0 = 0

function [A_solution, res, iter] = solpzy(A0, B0, err)
    max_iter = 30;
    iter = 0;   res = 10;
    while iter <= max_iter && res > err
        iter = iter + 1;
        % Define the objective function
        objective = @(A) abs(sum(sum(A .* log2(A))) - B0);
        % Define the constraints
        symmetryConstraint = @(A) norm(A - A', 'fro'); % Frobenius norm of A - A'
        rowSumConstraint = @(A) norm(sum(A, 2) - ones(size(A, 1), 1), 1); % L1 norm of row sums - 1
        columnSumConstraint = @(A) norm(sum(A, 1) - ones(1, size(A, 2)), 1); % L1 norm of column sums - 1

        % Combine constraints
        nonlinearconstr = @(A)deal([symmetryConstraint(A), rowSumConstraint(A), columnSumConstraint(A)], []);

        % Set options for the optimization
        options = optimoptions(@fmincon, 'Algorithm', 'interior-point', 'Display', 'iter');

        % Solve the optimization problem
        A_solution = fmincon(objective, A0, [], [], [], [], [], [], nonlinearconstr, options);
        A0 = A_solution;
        res = abs(objective(A0));
    end

end
