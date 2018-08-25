% calculate the TP index
function TP_rate = TP_calculation(I_Out,I_Mask)

cross_region = I_Out & I_Mask;
numerator = sum(sum(cross_region));
denominator = sum(sum(I_Mask));

TP_rate = numerator/denominator;



% TP_rate
% test_end = 1;