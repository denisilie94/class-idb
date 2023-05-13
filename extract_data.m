function [TrData, TtData, TrLabel, TtLabel] = extract_data(featureMat, labelMat, number, rng_seed)
    rng(rng_seed, 'twister');
    n_classes = size(labelMat, 1);
    n_features = size(featureMat, 1);

    TrData = zeros(n_features, number*n_classes);
    TtData = zeros(n_features, size(featureMat, 2) - number*n_classes);
    TrLabel = zeros(1, number*n_classes);
    TtLabel = zeros(1, size(featureMat, 2) - number*n_classes);
    
    i_tr = 1;
    i_tt = 1;

    for i_class = 1:n_classes
        TempData = featureMat(:, labelMat(i_class, :) == 1);
        TempLabel = i_class * ones(1, size(TempData, 2));

        l = length(TempLabel);
        index = randperm(l);

        TrData(:, i_tr:(i_tr+number-1)) = TempData(:, index(1:number));
        TrLabel(:,i_tr:(i_tr+number-1)) = TempLabel(:, index(1:number));
        TtData(:, i_tt:(i_tt+(l-number-1))) = TempData(:, index((number+1):end));
        TtLabel(:, i_tt:(i_tt+(l-number-1))) = TempLabel(:, index((number+1):end));

        i_tr = i_tr + number;
        i_tt = i_tt + l - number;
    end
end