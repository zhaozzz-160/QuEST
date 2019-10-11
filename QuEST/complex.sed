s/stateVec(Real|Imag)\s*(\[[^']']+\])/qureg.stateVec\2.\1/g;
 /qreal\s*\*stateVec/d;
 /shared/{s/\(/\(qureg, /;
s/(,)?\s*stateVec(Real|Imag)\s*//g};
  s/Real/real/g;
 s/Imag/imag/g;



