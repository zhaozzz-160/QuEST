s/(pairS|deviceS|s)tateVec\.(real|imag)(\[[^\x5D]+?\])/\1tateVec\3.\2/g;
s/(\w+S|s)tateVec(Real|Imag)(\[[^\x5D]+?\])/qureg.\1tateVec\3.\2/g;
s/(pairS|deviceS|s)tateVec(Real|Imag)\s*(\[[^\x5D]+?\])/qureg.\1tateVec\3.\2/g;
/qreal\s*\*stateVec/d;
s/ComplexArray\s*stateVec/Complex *stateVec/g;
s/stateVec(Real|Imag)(Up|Lo|Out)(\[[^\x5D]+?\])/stateVec\2\3.\1/g;
/shared.*(pairS|deviceS|s)tateVec(Real|Imag)(Up|Lo|Out)/{
    s/stateVec(Real|Imag)(Up|Lo|Out)/stateVec\2/g;
    s/(stateVec(Up|Lo|Out))\s*,\s*\1/\1/g};
/shared.*(pairS|deviceS|s)tateVec(Real|Imag)/{
    s/\(/\(qureg, /;s/(,)?\s*stateVec(Real|Imag)\s*//g};
s/\.Real/.real/g;
s/\.Imag/.imag/g;
