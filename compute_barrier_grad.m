function [g1, g2] = compute_barrier_grad(d, Dc, desired_coh)
    n_features = length(d);
    v = Dc'*d;
    w = max(abs(v)/desired_coh, 1);

    % what atoms are too close?
    i_minus = find(v>desired_coh);
    i_plus = find(v<-desired_coh);

    % gradients
    g1 = Dc*(w.*v);
    g2 = zeros(n_features, 1);
    if ~isempty(i_minus)
        g2 = g2 + sum(Dc(:,i_minus) - repmat(d,1,length(i_minus)), 2);
    end
    if ~isempty(i_plus)
        g2 = g2 - sum(Dc(:,i_plus) + repmat(d,1,length(i_plus)), 2);
    end
end

