function [grad] = compute_triplet_barrier_grad(d, D, Dc, M)
    sign_D = sign(d' * D);
    diff_D = d - sign_D .* D;
    l2_diff_D = vecnorm(diff_D).^2;

    sign_Dc = sign(d' * Dc);
    diff_Dc = d - sign_Dc .* Dc;
    l2_diff_Dc = vecnorm(diff_Dc).^2;

    r_diff_D = repelem(diff_D, 1, length(l2_diff_Dc));
    r_l2_diff_D = repelem(l2_diff_D, 1, length(l2_diff_Dc));
    r_diff_Dc = repmat(diff_Dc, 1, length(l2_diff_D));
    r_l2_diff_Dc = repmat(l2_diff_Dc, 1, length(l2_diff_D));

    v = max(0, M - r_l2_diff_D + r_l2_diff_Dc);
    index = find(v == 0);
    r_diff_D(:, index) = 0;
    r_diff_Dc(:, index) = 0;

    grad = sum(r_diff_D - r_diff_Dc, 2);
end

