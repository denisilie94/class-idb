function [accuracy, trainTime, testTime, D_all] = aksvd_idl(X_train, y_train,...
  X_test, y_test, gamma, n_components, n_nonzero_coefs, n_iterations, seed)

    % Prepare seed
    rng(seed)

    % Dataset properties
    n_classes = length(unique(y_train));

    X_all = cell(1, n_classes);
    D_all = cell(1, n_classes);
    for i_class = 1:n_classes
        D_all{i_class} = normcol_equal(randn(size(X_train,1), n_components));
    end

    % Start waitbar
    trainTime = 0;
    wb = waitbar(0, '[IDL] Training...');

    for i_iter = 1:n_iterations
        tmpTime = tic;
        for i_class = 1:n_classes
            % Coding method
            X_all{i_class} = omp(X_train(:, y_train==i_class), D_all{i_class}, n_nonzero_coefs);

            % Learning method
            Y = X_train(:, y_train==i_class);
            D = D_all{i_class};
            X = X_all{i_class};
            E = Y - D * X;

            Dc = [];
            for tmp_i_class = 1:n_classes
               if tmp_i_class ~= i_class
                  Dc = [Dc D_all{tmp_i_class}];
               end
            end

            for i_atom = 1:size(D, 2)
                [~, atom_usages, ~] = find(X(i_atom,:));

                if (isempty(atom_usages))
                    D(:, i_atom) = randn(size(D,1), 1);
                    D(:, i_atom) = D(:, i_atom) / norm(D(:, i_atom));
                else
                    F = E(:, atom_usages) + D(:, i_atom) * X(i_atom, atom_usages);
                    d = F * X(i_atom, atom_usages)' - 2 * gamma * Dc * (Dc' * D(:, i_atom));
                    D(:, i_atom) = d / norm(d);
                    X(i_atom, atom_usages) = F' * D(:, i_atom);
                    E(:, atom_usages) = F - D(:, i_atom) * X(i_atom, atom_usages);
                end
            end

            D_all{i_class} = D;
            X_all{i_class} = X;
        end
        trainTime = trainTime + toc(tmpTime);

        % Update waitbar
        waitbar(i_iter/n_iterations, wb, sprintf('[IDL] Training - Remaining time: %d [sec]',...
                round(trainTime/i_iter*(n_iterations - i_iter))));
    end

    % Close waitbar
    close(wb);


    % Start waitbar
    testTime = 0;
    wb = waitbar(0, '[IDL] Testing...');

    Errs = [];
    prediction = [];
    for i_test = 1:size(X_test, 2)

        tmpTime = tic;
        errs = [];
        for i_class = 1:n_classes
            x = omp(X_test(:, i_test), D_all{i_class}, n_nonzero_coefs);
            errs = [errs norm(X_test(:, i_test) - D_all{i_class} * x)];
        end

        Errs = [Errs; errs];
        [~, index] = min(errs);
        prediction = [prediction index];
        testTime = testTime + toc(tmpTime);

       % Update waitbar
       waitbar(i_test/size(X_test, 2), wb, sprintf('[IDL] Testing - Remaining time: %d [sec]',...
               round(testTime/i_test*(size(X_test, 2) - i_test))));
    end

    % Close waitbar
    close(wb);


    % Compute problem accuracy
    accuracy = sum(y_test==prediction)/size(X_test,2);
end


