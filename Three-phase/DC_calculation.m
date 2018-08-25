% calculate the DC index
function DC_rate = DC_calculation(I_Out,I_Mask)

cross_region = I_Out & I_Mask;
numerator = sum(sum(cross_region));
denominator = sum(sum(I_Out)) + sum(sum(I_Mask));
DC_rate = 2*numerator/denominator;


% TP_rate
% test_end = 1;