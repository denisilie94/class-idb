clc;
clear;
close all;

n_rounds = 1;
n_components_list = [40];
n_nonzero_coefs_prec_list = [0.5];
n_iterations_list = [10];
M_list = [1.2, 1.4, 1.6, 1.8];
gamma_list = [5e-8 5e-7 5e-6];% 5 5e+1 5e+2 5e+3];
lambda_list = [5e-5 5e-4 5e-3 5e-2 5e-1];% 5 5e+1 5e+2 5e+3];
triplet_prec_list = [0.05];

% this are used only for discrim dl
% alpha_list = logspace(-2, 2, 10);
% n_list = [1000];

rng_seed = 1;
yaleb_number = 32;
ar_face_number = 20;
caltech_101_number = 30;
scene_15_number = 30;
cmupie_number = 30;
ucf50_number = 30;
hmdb51_number = 30;

% DataPath = 'randomfaces4extendedyaleb'; number = yaleb_number;
% DataPath = 'randomfaces4ar'; number = ar_face_number;
DataPath = 'spatialpyramidfeatures4caltech101'; number = caltech_101_number;
% DataPath = 'spatialpyramidfeatures4scene15'; number = scene_15_number;
% DataPath = 'CMUPIE_random_256'; number = cmupie_number;
% DataPath = 'ucf50_dataset'; number = ucf50_number;
% DataPath = 'hmdb51_dataset'; number = hmdb51_number;
load(fullfile('dbs', DataPath));

[TrData, TtData, TrLabel, TtLabel] = extract_data(featureMat, labelMat,...
                                                  number, rng_seed);

% Data preprocessing
y_train = TrLabel;
y_test = TtLabel;
X_train = TrData;
X_test = TtData;


% AK-SVD IDL
fileID = fopen(sprintf('./logs/aksvd_%s_idl.log', DataPath),'a');

for seed = 1:n_rounds
    for i_n_components = 1:length(n_components_list)
        n_components = n_components_list(i_n_components);

        for i_n_nonzero_coefs_prec = 1:length(n_nonzero_coefs_prec_list)
            n_nonzero_coefs_prec = n_nonzero_coefs_prec_list(i_n_nonzero_coefs_prec);
            n_nonzero_coefs = round(n_nonzero_coefs_prec * n_components);

            for i_n_iterations = 1:length(n_iterations_list)
                n_iterations = n_iterations_list(i_n_iterations);

                for i_gamma = 1:length(gamma_list)
                    gamma = gamma_list(i_gamma);

                    [accuracy, trainTime, testTime, D_all] = aksvd_idl(X_train, y_train,...
                        X_test, y_test, gamma, n_components, n_nonzero_coefs, n_iterations, seed);

                    mat_file = sprintf('idl_n_components_%d_n_nonzero_coefs_%d_n_iterations_%d_gamma_%d.mat',...
                                       n_components, n_nonzero_coefs, n_iterations, gamma);
%                     save(['./mats/' mat_file], 'DataPath', 'accuracy', 'trainTime', 'testTime', 'D_all',...
%                          'n_components', 'n_nonzero_coefs', 'n_iterations', 'gamma');

                    fprintf(fileID, '%s: %f\n', mat_file, accuracy);
                end
            end
        end 
    end
end
    
fclose(fileID);




% AK-SVD IDB
fileID = fopen(sprintf('./logs/aksvd_%s_idb.log', DataPath),'a');

for seed = 1:n_rounds
    for i_n_components = 1:length(n_components_list)
        n_components = n_components_list(i_n_components);

        for i_n_nonzero_coefs_prec = 1:length(n_nonzero_coefs_prec_list)
            n_nonzero_coefs_prec = n_nonzero_coefs_prec_list(i_n_nonzero_coefs_prec);
            n_nonzero_coefs = round(n_nonzero_coefs_prec * n_components);

            for i_n_iterations = 1:length(n_iterations_list)
                n_iterations = n_iterations_list(i_n_iterations);

                for i_M = 1:length(M_list)
                    M = M_list(i_M);
                
                    for i_gamma = 1:length(gamma_list)
                        gamma = gamma_list(i_gamma);

                        for i_lambda = 1:length(lambda_list)
                            lambda = lambda_list(i_lambda);

                            [accuracy, trainTime, testTime, D_all] = aksvd_idb(X_train, y_train,...
                                X_test, y_test, n_components, n_nonzero_coefs, n_iterations, M, gamma,...
                                lambda, seed);

                            mat_file = sprintf('idb_n_components_%d_n_nonzero_coefs_%d_n_iterations_%d_gamma_%f_lambda_%f_M_%d.mat',...
                                               n_components, n_nonzero_coefs, n_iterations, gamma, lambda, M);
%                             save(['./mats/' mat_file], 'DataPath', 'accuracy', 'trainTime', 'testTime', 'D_all',...
%                                  'n_components', 'n_nonzero_coefs', 'n_iterations', 'gamma', 'lambda', 'M');

                            fprintf(fileID, '%s: %f\n', mat_file, accuracy);
                        end
                    end
                end
            end
        end 
    end
end

fclose(fileID);




% AK-SVD ITDB
fileID = fopen(sprintf('./logs/aksvd_%s_itdb.log', DataPath),'a');

for seed = 1:n_rounds
    for i_n_components = 1:length(n_components_list)
        n_components = n_components_list(i_n_components);

        for i_n_nonzero_coefs_prec = 1:length(n_nonzero_coefs_prec_list)
            n_nonzero_coefs_prec = n_nonzero_coefs_prec_list(i_n_nonzero_coefs_prec);
            n_nonzero_coefs = round(n_nonzero_coefs_prec * n_components);

            for i_n_iterations = 1:length(n_iterations_list)
                n_iterations = n_iterations_list(i_n_iterations);

                for i_M = 1:length(M_list)
                    M = M_list(i_M);
                
                    for i_gamma = 1:length(gamma_list)
                        gamma = gamma_list(i_gamma);

                        for i_triple_prec = 1:length(triplet_prec_list)
                            triplet_prec = triplet_prec_list(i_triple_prec);

                            [accuracy, trainTime, testTime, D_all] = aksvd_itdb(X_train, y_train,...
                                X_test, y_test, n_components, n_nonzero_coefs, n_iterations, M, gamma,...
                                triplet_prec, seed);

                            mat_file = sprintf('itdb_n_components_%d_n_nonzero_coefs_%d_n_iterations_%d_gamma_%f_triplet_prec_%f_M_%d.mat',...
                                               n_components, n_nonzero_coefs, n_iterations, gamma, triplet_prec, M);
%                             save(['./mats/' mat_file], 'DataPath', 'accuracy', 'trainTime', 'testTime', 'D_all',...
%                                  'n_components', 'n_nonzero_coefs', 'n_iterations', 'gamma', 'triplet_prec', 'M');

                            fprintf(fileID, '%s: %f\n', mat_file, accuracy);
                        end
                    end
                end
            end
        end 
    end
end

fclose(fileID);
