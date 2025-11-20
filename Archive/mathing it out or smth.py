#MATH PART!! for force
import math

mA = 2.0 
mB = 3.0  
viA = 5.0 
viB = 1.0 
C = 0.8  

sqrtt = math.sqrt(
    mA*mB*
    ((C * mA**2 * viA**2) +
    (C * mA * mB * viA**2) +
    (C * mB**2 * viB**2) +
    (C * mA * mB * viB**2) -
    (mA**2 * viA**2) -
    (2 * mA * mB * viA * viB) -
    (mB**2 * viB**2))
)

v1 = (-sqrtt + mA**2 * viA + mA * mB * viB) / (mA * (mA + mB))
v2 = ( sqrtt + mA * mB * viA + mB**2 * viB) / (mB * (mA + mB))



#fij calculation 











