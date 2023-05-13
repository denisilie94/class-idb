clc;
clear;
close all;

n_rounds = 10;
n_components = 40;
n_nonzero_coefs = 20;
n_iterations = 10;
file = fopen('logs_idb.txt', 'a');

yaleb_number = 32;
ar_face_number = 20;
caltech_101_number = 30;
scene_15_number = 30;
cmupie_number = 30;
ucf50_number = 30;
hmdb51_number = 30;

DataPaths = [
%     "randomfaces4extendedyaleb",...
%     "randomfaces4ar",...
%     "spatialpyramidfeatures4caltech101",...
%     "spatialpyramidfeatures4scene15",...
%     "CMUPIE_random_256",...
%     "ucf50_dataset"...
    "hmdb51_dataset"
];

samples_numbers = [
%     yaleb_number,...
%     ar_face_number,...
%     caltech_101_number,...
%     scene_15_number,...
%     cmupie_number,...
%     ucf50_number,...
    hmdb51_number
];

for i_DataPath = 1:length(DataPaths)
    DataPath = DataPaths(i_DataPath);
    number = samples_numbers(i_DataPath);
    load(fullfile('./dbs', DataPath));
    
    % Choose parameters by db
    switch DataPath
        case 'randomfaces4extendedyaleb'
            M = 1.4;
            lambda = 500;
            gamma = 0.05;
        case 'randomfaces4ar'
            M = 1.2;
            lambda = 0.5;
            gamma = 5000;
        case 'spatialpyramidfeatures4caltech101'   
            M = 1.2;
            lambda = 0.5;
            gamma = 0.005;
        case 'ucf50_dataset'
            M = 1.2;
            lambda = 500;
            gamma = 500;
        case 'CMUPIE_random_256'
            M = 1.4;
            lambda = 500;
            gamma = 0.5;
        case 'spatialpyramidfeatures4scene15'
            M = 1.4;
            lambda = 0.005;
            gamma = 0.005;
        case 'hmdb51_dataset'
            M = 1.2;
            lambda = 5000;
            gamma = 500;
    end
    
    T_Accuracy = 0;
    T_TrTime = 0;
    T_TtTime = 0;
    
    for i_round = 1:n_rounds
        [TrData, TtData, TrLabel, TtLabel] = extract_data(featureMat, labelMat,number, i_round);

        % Data preprocessing
        y_train = TrLabel;
        y_test = TtLabel;
        X_train = TrData;
        X_test = TtData;

        [accuracy, trainTime, testTime, D_all] = aksvd_idb(X_train, y_train,...
          X_test, y_test, n_components, n_nonzero_coefs, n_iterations, M, gamma, lambda,...
          i_round);

        fprintf('[IDB] Accuracy: %f\n', accuracy);
        fprintf('[IDB] Training time: %f [sec]\n', trainTime);
        fprintf('[IDB] Testing time: %f [sec]\n', testTime);
        fprintf('\n');

        fprintf(file, '[IDB] Accuracy: %f\n', accuracy);
        fprintf(file, '[IDB] Training time: %f [sec]\n', trainTime);
        fprintf(file, '[IDB] Testing time: %f [sec]\n', testTime);
        fprintf(file, '\n');
        
        T_Accuracy = T_Accuracy + accuracy;
        T_TrTime = T_TrTime + trainTime;
        T_TtTime = T_TtTime + testTime;
    end
    
    T_Accuracy = T_Accuracy / n_rounds;
    T_TrTime = T_TrTime / n_rounds;
    T_TtTime = T_TtTime / n_rounds;
    
    fprintf(file, '\n%s\n', DataPath);
    fprintf(file, 'The running time for IDB training is : %.03f\n', T_TrTime);
    fprintf(file, 'The running time for IDB testing is : %.03f\n', T_TtTime);
    fprintf(file, 'Recognition rate for IDB is : %.04f\n', T_Accuracy);
    fprintf(file, '\n\n');
    fprintf(file, '======================================================');
    fprintf(file, '\n\n');

end

fclose(file);