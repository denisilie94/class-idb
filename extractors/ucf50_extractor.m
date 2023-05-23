% This script is used to extract features from the
% original repository of action bank from Buffalo University.

clc;
clear;

k = 5000; % number of pca features
dirFolders = dir('../data/ab_ucf50_matlab');

position = 1;
label = 1;
db_features = zeros(14965, 6680);

for i = 1:length(dirFolders)
   if ~strcmp(dirFolders(i).name, '.') && ~strcmp(dirFolders(i).name, '..') && dirFolders(i).isdir
       files = dir(strcat(dirFolders(i).folder, '/', dirFolders(i).name, '/*.mat'));
       
       for j = 1:length(files)
            tmp_data = load(strcat(files(j).folder, '/', files(j).name));
            db_features(:, position) = tmp_data.v;
            db_labels(position) = label;
            position = position + 1;
       end
       
       label = label + 1;
   end
end

[coeff, ~, ~] = pca(db_features');
featureMat = (db_features' * coeff(:, 1:k))';
n_samples = size(db_features, 2);
n_classes = length(unique(db_labels));
labelMat = zeros(n_classes, n_samples);

for i = 1:n_samples
    labelMat(db_labels(i), i) = 1;
end

save('ucf50_dataset.mat', 'featureMat', 'labelMat');