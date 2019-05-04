function B = Benefit(Y, Y_predicted)
   % Assuming larger labels more desirable.
    B = Y_predicted - Y + 1;
    %
    %epsilon = 1;
    %B = (Y_predicted + epsilon)./(Y + epsilon) + epsilon;   
    %
    %B = exp(Y_predicted - Y);
end