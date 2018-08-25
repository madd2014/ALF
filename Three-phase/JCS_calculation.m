% calculate the JCS index
function JCS_rate = JCS_calculation(I_Out,I_Mask)

cross_region = I_Out & I_Mask;
numerator = sum(sum(cross_region));
unit_region = I_Out | I_Mask;
denominator = sum(sum(unit_region));

JCS_rate = numerator/denominator;

% TP_rate
% test_end = 1;