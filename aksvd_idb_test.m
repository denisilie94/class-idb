clc;
clear;
close all;

n_components = 40;
n_nonzero_coefs = 20;
n_iterations = 10;

M = 1.2;                % barrier margin
gamma = 0.05; %0.5;            % trade-off factors
lambda = 1E+2;%25;
desired_coh = 1 - M/2;

rng_seed = 1;
yaleb_number = 32;
ar_face_number = 20;
caltech_101_number = 30;
scene_15_number = 30;
cmupie_number = 30;
hmdb51_number = 30;

DataPath = 'randomfaces4extendedyaleb'; number = yaleb_number;
% DataPath = 'randomfaces4ar'; number = ar_face_number;
% DataPath = 'spatialpyramidfeatures4caltech101'; number = caltech_101_number;
% DataPath = 'spatialpyramidfeatures4scene15'; number = scene_15_number;
% DataPath = 'CMUPIE_random_256'; number = cmupie_number;
% DataPath = 'hmdb51_dataset'; number = hmdb51_number;
load(fullfile('dbs', DataPath));

[TrData, TtData, TrLabel, TtLabel] = extract_data(featureMat, labelMat,...
                                                  number, rng_seed);

% Data preprocessing
y_train = TrLabel;
y_test = TtLabel;
X_train = TrData;
X_test = TtData;

[accuracy, trainTime, testTime, D_all] = aksvd_idb(X_train, y_train,...
  X_test, y_test, n_components, n_nonzero_coefs, n_iterations, M, gamma, lambda,...
  'default');

fprintf("[IDB] Accuracy: %f\n", accuracy);
fprintf('[IDB] Training time: %f [sec]\n', trainTime);
fprintf('[IDB] Testing time: %f [sec]\n', testTime);
fprintf('\n');

