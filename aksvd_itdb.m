function [accuracy, trainTime, testTime, D_all] = aksvd_itdb(X_train, y_train,...
    X_test, y_test, n_components, n_nonzero_coefs, n_iterations, M, gamma,...
    triplet_prec, seed)

    % Prepare seed
    rng(seed)

    % Dataset properties
    n_classes = length(unique(y_train));
    n_features = size(X_train, 1);

    X_all = cell(1, n_classes);
    D_all = cell(1, n_classes);
    for i_class = 1:n_classes
        D_all{i_class} = normcol_equal(randn(size(X_train,1), n_components));
    end


    % Start waitbar
    trainTime = 0;
    wb = waitbar(0, '[ITDB] Training...');

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

            for j = 1:n_components
                [~, data_indices, x] = find(X(j,:));

                if (isempty(data_indices))
                    d = randn(n_features, 1);
                    D(:, j) = d / norm(d);
                else
                    d = D(:, j);
                    rp = randperm(size(Dc, 2));
                    rp = rp(1:length(rp) * triplet_prec);
                    grad = compute_triplet_barrier_grad(d, D, Dc(:, rp), M);

                    F = E(:, data_indices) + d * x;
                    d = F*x' - gamma*grad;

                    D(:, j) = d / norm(d);
                    X(j, data_indices) = F'*D(:, j);
                    E(:, data_indices) = F - D(:, j)*X(j, data_indices);
                end
            end

            D_all{i_class} = D;
            X_all{i_class} = X;
        end
        trainTime = trainTime + toc(tmpTime);

        % Update waitbar
        waitbar(i_iter/n_iterations, wb, sprintf('[ITDB] Training - Remaining time: %d [sec]',...
                round(trainTime/i_iter*(n_iterations - i_iter))));




%         if mod(i_iter, 5) == 0
%             Errs = [];
%             prediction = [];
%             for i_test = 1:size(X_test, 2)
%                 errs = [];
%                 for i_class = 1:n_classes
%                     x = omp(X_test(:, i_test), D_all{i_class}, n_nonzero_coefs);
%                     errs = [errs norm(X_test(:, i_test) - D_all{i_class} * x)];
%                 end
% 
%                 Errs = [Errs; errs];
%                 [~, index] = min(errs);
%                 prediction = [prediction index];
%             end
% 
%             % Compute problem accuracy
%             accuracy = sum(y_test==prediction)/size(X_test,2);
%             fprintf("[ITDB] Accuracy round %d: %f\n", i_iter, accuracy);
%             fprintf('\n');
%         end
    end

    % Close waitbar
    close(wb);


    % Start waitbar
    testTime = 0;
    wb = waitbar(0, '[ITDB] Testing...');

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
       waitbar(i_test/size(X_test, 2), wb, sprintf('[ITDB] Testing - Remaining time: %d [sec]',...
               round(testTime/i_test*(size(X_test, 2) - i_test))));
    end

    % Close waitbar
    close(wb);


    % Compute problem accuracy
    accuracy = sum(y_test==prediction)/size(X_test,2);

end
