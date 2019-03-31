function FP_rate = FP_calculation(I_Out,I_Mask)

unit_region = I_Out | I_Mask;
numerator = sum(sum(unit_region - I_Mask));
denominator = sum(sum(I_Mask));
FP_rate = numerator/denominator;

