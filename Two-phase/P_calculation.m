% calculate the Precision index
function P_rate = P_calculation(I_Out,I_Mask)

cross_region = I_Out & I_Mask;
numerator = sum(sum(cross_region));
denominator = sum(sum(I_Mask));
TP_rate = numerator/denominator;

unit_region = I_Out | I_Mask;
numerator = sum(sum(unit_region - I_Mask));
denominator = sum(sum(I_Mask));
FP_rate = numerator/denominator;

P_rate = TP_rate/(TP_rate + FP_rate);

% TP_rate
% test_end = 1;