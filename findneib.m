% Find the neighbors of the matrix A in a restricted region eps according to the Frobenius norm

function [matrix_neib] = findneib(A, eps, num_neib)
    min_value = -eps;   max_value = eps;
    % Create a grid of points in the matrix space
    [X, Y, Z, W] = ndgrid(min_value:eps:max_value, min_value:eps:max_value, min_value:eps:max_value, min_value:eps:max_value);

    % Iterate over each quadruplet of elements in the grid and create matrices
    num_elements = numel(X);
    matrices = cell(num_elements, 1);
    for i = 1:num_elements
        matrices{i} = A + [X(i), Y(i), Z(i), W(i); X(i), Y(i), Z(i), W(i); X(i), Y(i), Z(i), W(i); X(i), Y(i), Z(i), W(i)];
    end

    % Compute the Frobenius norm between A and each matrix in the grid
    frobenius_norms = zeros(num_elements, 1);
    for i = 1:num_elements
        frobenius_norms(i) = norm(matrices{i} - A, 'fro');
    end

    % Sort the matrices based on their Frobenius norm values
    [~, sorted_indices] = sort(-frobenius_norms);
    sorted_matrices = matrices(sorted_indices);

    matrix_neib = cell(num_neib, 1);
    for i = 1:num_neib
        matrix_neib{i} = sorted_matrices{i};
    end
end

