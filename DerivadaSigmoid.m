function [D] = DerivadaSigmoid(x)
    D = x.*(1-x);
end