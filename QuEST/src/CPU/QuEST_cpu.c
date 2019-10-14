// Distributed under MIT licence. See https://github.com/QuEST-Kit/QuEST/blob/master/LICENCE.txt for details

/** @file
 * The core of the CPU backend functionality. The CPU/MPI implementations of the pure state functions in
 * ../QuEST_ops_pure.h are in QuEST_cpu_local.c and QuEST_cpu_distributed.c which mostly wrap the core
 * functions defined here. Some additional hardware-agnostic functions are defined here
 */

# include "QuEST.h"
# include "QuEST_internal.h"
# include "QuEST_precision.h"
# include "mt19937ar.h"

# include "QuEST_cpu_internal.h"

# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <assert.h>

# ifdef _OPENMP
# include <omp.h>
# endif

/** Get the value of the bit at a particular index in a number.
  SCB edit: new definition of extractBit is much faster ***
 * @param[in] locationOfBitFromRight location of bit in theEncodedNumber
 * @param[in] theEncodedNumber number to search
 * @return the value of the bit in theEncodedNumber
 */
static int extractBit (const int locationOfBitFromRight, const long long int theEncodedNumber)
{
    return (theEncodedNumber & ( 1LL << locationOfBitFromRight )) >> locationOfBitFromRight;
}

void densmatr_oneQubitDegradeOffDiagonal(Qureg qureg, const int targetQubit, qreal retain){
    const long long int numTasks = qureg.numAmpsPerChunk;
    long long int innerMask = 1LL << targetQubit;
    long long int outerMask = 1LL << (targetQubit + (qureg.numQubitsRepresented));

    long long int thisTask;
    long long int thisPattern;
    long long int totMask = innerMask|outerMask;

 # ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (innerMask,outerMask,totMask,qureg,retain) \
    private  (thisTask,thisPattern)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++){
            thisPattern = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMask;
            if ((thisPattern==innerMask) || (thisPattern==outerMask)){
                // do dephase
                // the lines below will degrade the off-diagonal terms |..0..><..1..| and |..1..><..0..|
                qureg.stateVec[thisTask].real = retain*qureg.stateVec[thisTask].real;
                qureg.stateVec[thisTask].imag = retain*qureg.stateVec[thisTask].imag;
            }
        }
    }
}

void densmatr_oneQubitDephase(Qureg qureg, const int targetQubit, qreal dephase) {
    qreal retain=1-dephase;
    densmatr_oneQubitDegradeOffDiagonal(qureg, targetQubit, retain);
}

void densmatr_oneQubitDampingDephase(Qureg qureg, const int targetQubit, qreal dephase) {
    qreal retain=sqrt(1-dephase);
    densmatr_oneQubitDegradeOffDiagonal(qureg, targetQubit, retain);
}

void densmatr_twoQubitDephase(Qureg qureg, const int qubit1, const int qubit2, qreal dephase) {
    qreal retain=1-dephase;

    const long long int numTasks = qureg.numAmpsPerChunk;
    long long int innerMaskQubit1 = 1LL << qubit1;
    long long int outerMaskQubit1 = 1LL << (qubit1 + (qureg.numQubitsRepresented));
    long long int innerMaskQubit2 = 1LL << qubit2;
    long long int outerMaskQubit2 = 1LL << (qubit2 + (qureg.numQubitsRepresented));
    long long int totMaskQubit1 = innerMaskQubit1|outerMaskQubit1;
    long long int totMaskQubit2 = innerMaskQubit2|outerMaskQubit2;

    long long int thisTask;
    long long int thisPatternQubit1, thisPatternQubit2;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (innerMaskQubit1,outerMaskQubit1,totMaskQubit1,innerMaskQubit2,outerMaskQubit2, \
                totMaskQubit2,qureg,retain) \
    private  (thisTask,thisPatternQubit1,thisPatternQubit2)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++){
            thisPatternQubit1 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit1;
            thisPatternQubit2 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit2;

            // any mismatch |...0...><...1...| etc
            if ( (thisPatternQubit1==innerMaskQubit1) || (thisPatternQubit1==outerMaskQubit1) ||
                    (thisPatternQubit2==innerMaskQubit2) || (thisPatternQubit2==outerMaskQubit2) ){
                // do dephase
                // the lines below will degrade the off-diagonal terms |..0..><..1..| and |..1..><..0..|
                qureg.stateVec[thisTask].real = retain*qureg.stateVec[thisTask].real;
                qureg.stateVec[thisTask].imag = retain*qureg.stateVec[thisTask].imag;
            }
        }
    }
}

void densmatr_oneQubitDepolariseLocal(Qureg qureg, const int targetQubit, qreal depolLevel) {
    qreal retain=1-depolLevel;

    const long long int numTasks = qureg.numAmpsPerChunk;
    long long int innerMask = 1LL << targetQubit;
    long long int outerMask = 1LL << (targetQubit + (qureg.numQubitsRepresented));
    long long int totMask = innerMask|outerMask;

    long long int thisTask;
    long long int partner;
    long long int thisPattern;

    qreal realAv, imagAv;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (innerMask,outerMask,totMask,qureg,retain,depolLevel) \
    private  (thisTask,partner,thisPattern,realAv,imagAv)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++){
            thisPattern = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMask;
            if ((thisPattern==innerMask) || (thisPattern==outerMask)){
                // do dephase
                // the lines below will degrade the off-diagonal terms |..0..><..1..| and |..1..><..0..|
                qureg.stateVec[thisTask].real = retain*qureg.stateVec[thisTask].real;
                qureg.stateVec[thisTask].imag = retain*qureg.stateVec[thisTask].imag;
            } else {
                if ((thisTask&totMask)==0){ //this element relates to targetQubit in state 0
                    // do depolarise
                    partner = thisTask | totMask;
                    realAv =  (qureg.stateVec[thisTask].real + qureg.stateVec[partner].real) /2 ;
                    imagAv =  (qureg.stateVec[thisTask].imag + qureg.stateVec[partner].imag) /2 ;

                    qureg.stateVec[thisTask].real = retain*qureg.stateVec[thisTask].real + depolLevel*realAv;
                    qureg.stateVec[thisTask].imag = retain*qureg.stateVec[thisTask].imag + depolLevel*imagAv;

                    qureg.stateVec[partner].real = retain*qureg.stateVec[partner].real + depolLevel*realAv;
                    qureg.stateVec[partner].imag = retain*qureg.stateVec[partner].imag + depolLevel*imagAv;
                }
            }
        }
    }
}

void densmatr_oneQubitDampingLocal(Qureg qureg, const int targetQubit, qreal damping) {
    qreal retain=1-damping;
    qreal dephase=sqrt(retain);

    const long long int numTasks = qureg.numAmpsPerChunk;
    long long int innerMask = 1LL << targetQubit;
    long long int outerMask = 1LL << (targetQubit + (qureg.numQubitsRepresented));
    long long int totMask = innerMask|outerMask;

    long long int thisTask;
    long long int partner;
    long long int thisPattern;

    //qreal realAv, imagAv;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (innerMask,outerMask,totMask,qureg,retain,damping,dephase) \
    private  (thisTask,partner,thisPattern)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++){
            thisPattern = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMask;
            if ((thisPattern==innerMask) || (thisPattern==outerMask)){
                // do dephase
                // the lines below will degrade the off-diagonal terms |..0..><..1..| and |..1..><..0..|
                qureg.stateVec[thisTask].real = dephase*qureg.stateVec[thisTask].real;
                qureg.stateVec[thisTask].imag = dephase*qureg.stateVec[thisTask].imag;
            } else {
                if ((thisTask&totMask)==0){ //this element relates to targetQubit in state 0
                    // do depolarise
                    partner = thisTask | totMask;
                    //realAv =  (qureg.stateVec[thisTask].real + qureg.stateVec[partner].real) /2 ;
                    //imagAv =  (qureg.stateVec[thisTask].imag + qureg.stateVec[partner].imag) /2 ;

                    qureg.stateVec[thisTask].real = qureg.stateVec[thisTask].real + damping*qureg.stateVec[partner].real;
                    qureg.stateVec[thisTask].imag = qureg.stateVec[thisTask].imag + damping*qureg.stateVec[partner].imag;

                    qureg.stateVec[partner].real = retain*qureg.stateVec[partner].real;
                    qureg.stateVec[partner].imag = retain*qureg.stateVec[partner].imag;
                }
            }
        }
    }
}

void densmatr_oneQubitDepolariseDistributed(Qureg qureg, const int targetQubit, qreal depolLevel) {

    // first do dephase part.
    // TODO -- this might be more efficient to do at the same time as the depolarise if we move to
    // iterating over all elements in the state vector for the purpose of vectorisation
    // TODO -- if we keep this split, move this function to densmatr_oneQubitDepolarise()
    densmatr_oneQubitDephase(qureg, targetQubit, depolLevel);

    long long int sizeInnerBlock, sizeInnerHalfBlock;
    long long int sizeOuterColumn, sizeOuterHalfColumn;
    long long int thisInnerBlock, // current block
         thisOuterColumn, // current column in density matrix
         thisIndex,    // current index in (density matrix representation) state vector
         thisIndexInOuterColumn,
         thisIndexInInnerBlock;
    int outerBit;

    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    // set dimensions
    sizeInnerHalfBlock = 1LL << targetQubit;
    sizeInnerBlock = 2LL * sizeInnerHalfBlock;
    sizeOuterColumn = 1LL << qureg.numQubitsRepresented;
    sizeOuterHalfColumn = sizeOuterColumn >> 1;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (sizeInnerBlock,sizeInnerHalfBlock,sizeOuterColumn,sizeOuterHalfColumn,qureg,depolLevel) \
    private  (thisTask,thisInnerBlock,thisOuterColumn,thisIndex,thisIndexInOuterColumn, \
                thisIndexInInnerBlock,outerBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        // thisTask iterates over half the elements in this process' chunk of the density matrix
        // treat this as iterating over all columns, then iterating over half the values
        // within one column.
        // If this function has been called, this process' chunk contains half an
        // outer block or less
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            // we want to process all columns in the density matrix,
            // updating the values for half of each column (one half of each inner block)
            thisOuterColumn = thisTask / sizeOuterHalfColumn;
            thisIndexInOuterColumn = thisTask&(sizeOuterHalfColumn-1); // thisTask % sizeOuterHalfColumn
            thisInnerBlock = thisIndexInOuterColumn/sizeInnerHalfBlock;
            // get index in state vector corresponding to upper inner block
            thisIndexInInnerBlock = thisTask&(sizeInnerHalfBlock-1); // thisTask % sizeInnerHalfBlock
            thisIndex = thisOuterColumn*sizeOuterColumn + thisInnerBlock*sizeInnerBlock
                + thisIndexInInnerBlock;
            // check if we are in the upper or lower half of an outer block
            outerBit = extractBit(targetQubit, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId)>>qureg.numQubitsRepresented);
            // if we are in the lower half of an outer block, shift to be in the lower half
            // of the inner block as well (we want to dephase |0><0| and |1><1| only)
            thisIndex += outerBit*(sizeInnerHalfBlock);

            // NOTE: at this point thisIndex should be the index of the element we want to
            // dephase in the chunk of the state vector on this process, in the
            // density matrix representation.
            // thisTask is the index of the pair element in pairStateVec


            // state[thisIndex] = (1-depolLevel)*state[thisIndex] + depolLevel*(state[thisIndex]
            //      + pair[thisTask])/2
            qureg.stateVec[thisIndex].real = (1-depolLevel)*qureg.stateVec[thisIndex].real +
                    depolLevel*(qureg.stateVec[thisIndex].real + qureg.pairStateVec[thisTask].real)/2;

            qureg.stateVec[thisIndex].imag = (1-depolLevel)*qureg.stateVec[thisIndex].imag +
                    depolLevel*(qureg.stateVec[thisIndex].imag + qureg.pairStateVec[thisTask].imag)/2;
        }
    }
}

void densmatr_oneQubitDampingDistributed(Qureg qureg, const int targetQubit, qreal damping) {
    qreal retain=1-damping;
    qreal dephase=sqrt(1-damping);
    // first do dephase part.
    // TODO -- this might be more efficient to do at the same time as the depolarise if we move to
    // iterating over all elements in the state vector for the purpose of vectorisation
    // TODO -- if we keep this split, move this function to densmatr_oneQubitDepolarise()
    densmatr_oneQubitDampingDephase(qureg, targetQubit, dephase);

    long long int sizeInnerBlock, sizeInnerHalfBlock;
    long long int sizeOuterColumn, sizeOuterHalfColumn;
    long long int thisInnerBlock, // current block
         thisOuterColumn, // current column in density matrix
         thisIndex,    // current index in (density matrix representation) state vector
         thisIndexInOuterColumn,
         thisIndexInInnerBlock;
    int outerBit;
    int stateBit;

    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    // set dimensions
    sizeInnerHalfBlock = 1LL << targetQubit;
    sizeInnerBlock = 2LL * sizeInnerHalfBlock;
    sizeOuterColumn = 1LL << qureg.numQubitsRepresented;
    sizeOuterHalfColumn = sizeOuterColumn >> 1;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (sizeInnerBlock,sizeInnerHalfBlock,sizeOuterColumn,sizeOuterHalfColumn,qureg,damping, retain, dephase) \
    private  (thisTask,thisInnerBlock,thisOuterColumn,thisIndex,thisIndexInOuterColumn, \
                thisIndexInInnerBlock,outerBit, stateBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        // thisTask iterates over half the elements in this process' chunk of the density matrix
        // treat this as iterating over all columns, then iterating over half the values
        // within one column.
        // If this function has been called, this process' chunk contains half an
        // outer block or less
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            // we want to process all columns in the density matrix,
            // updating the values for half of each column (one half of each inner block)
            thisOuterColumn = thisTask / sizeOuterHalfColumn;
            thisIndexInOuterColumn = thisTask&(sizeOuterHalfColumn-1); // thisTask % sizeOuterHalfColumn
            thisInnerBlock = thisIndexInOuterColumn/sizeInnerHalfBlock;
            // get index in state vector corresponding to upper inner block
            thisIndexInInnerBlock = thisTask&(sizeInnerHalfBlock-1); // thisTask % sizeInnerHalfBlock
            thisIndex = thisOuterColumn*sizeOuterColumn + thisInnerBlock*sizeInnerBlock
                + thisIndexInInnerBlock;
            // check if we are in the upper or lower half of an outer block
            outerBit = extractBit(targetQubit, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId)>>qureg.numQubitsRepresented);
            // if we are in the lower half of an outer block, shift to be in the lower half
            // of the inner block as well (we want to dephase |0><0| and |1><1| only)
            thisIndex += outerBit*(sizeInnerHalfBlock);

            // NOTE: at this point thisIndex should be the index of the element we want to
            // dephase in the chunk of the state vector on this process, in the
            // density matrix representation.
            // thisTask is the index of the pair element in pairStateVec

            // Extract state bit, is 0 if thisIndex corresponds to a state with 0 in the target qubit
            // and is 1 if thisIndex corresponds to a state with 1 in the target qubit
            stateBit = extractBit(targetQubit, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId));

            // state[thisIndex] = (1-depolLevel)*state[thisIndex] + depolLevel*(state[thisIndex]
            //      + pair[thisTask])/2
            if(stateBit == 0){
                qureg.stateVec[thisIndex].real = qureg.stateVec[thisIndex].real +
                    damping*( qureg.pairStateVec[thisTask].real);

                qureg.stateVec[thisIndex].imag = qureg.stateVec[thisIndex].imag +
                    damping*( qureg.pairStateVec[thisTask].imag);
            } else{
                qureg.stateVec[thisIndex].real = retain*qureg.stateVec[thisIndex].real;

                qureg.stateVec[thisIndex].imag = retain*qureg.stateVec[thisIndex].imag;
            }
        }
    }
}

// @TODO
void densmatr_twoQubitDepolariseLocal(Qureg qureg, int qubit1, int qubit2, qreal delta, qreal gamma) {
    const long long int numTasks = qureg.numAmpsPerChunk;
    long long int innerMaskQubit1 = 1LL << qubit1;
    long long int outerMaskQubit1= 1LL << (qubit1 + qureg.numQubitsRepresented);
    long long int totMaskQubit1 = innerMaskQubit1 | outerMaskQubit1;
    long long int innerMaskQubit2 = 1LL << qubit2;
    long long int outerMaskQubit2 = 1LL << (qubit2 + qureg.numQubitsRepresented);
    long long int totMaskQubit2 = innerMaskQubit2 | outerMaskQubit2;

    long long int thisTask;
    long long int partner;
    long long int thisPatternQubit1, thisPatternQubit2;

    qreal real00, imag00;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (totMaskQubit1,totMaskQubit2,qureg,delta,gamma) \
    private  (thisTask,partner,thisPatternQubit1,thisPatternQubit2,real00,imag00)
# endif
    {

# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        //--------------------------------------- STEP ONE ---------------------
        for (thisTask=0; thisTask<numTasks; thisTask++){
            thisPatternQubit1 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit1;
            thisPatternQubit2 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit2;
            if ((thisPatternQubit1==0) && ((thisPatternQubit2==0)
                        || (thisPatternQubit2==totMaskQubit2))){
                //this element of form |...X...0...><...X...0...|  for X either 0 or 1.
                partner = thisTask | totMaskQubit1;
                real00 =  qureg.stateVec[thisTask].real;
                imag00 =  qureg.stateVec[thisTask].imag;

                qureg.stateVec[thisTask].real = qureg.stateVec[thisTask].real
                    + delta*qureg.stateVec[partner].real;
                qureg.stateVec[thisTask].imag = qureg.stateVec[thisTask].imag
                    + delta*qureg.stateVec[partner].imag;

                qureg.stateVec[partner].real = qureg.stateVec[partner].real + delta*real00;
                qureg.stateVec[partner].imag = qureg.stateVec[partner].imag + delta*imag00;

            }
        }
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        //--------------------------------------- STEP TWO ---------------------
        for (thisTask=0; thisTask<numTasks; thisTask++){
            thisPatternQubit1 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit1;
            thisPatternQubit2 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit2;
            if ((thisPatternQubit2==0) && ((thisPatternQubit1==0)
                        || (thisPatternQubit1==totMaskQubit1))){
                //this element of form |...0...X...><...0...X...|  for X either 0 or 1.
                partner = thisTask | totMaskQubit2;
                real00 =  qureg.stateVec[thisTask].real;
                imag00 =  qureg.stateVec[thisTask].imag;

                qureg.stateVec[thisTask].real = qureg.stateVec[thisTask].real
                    + delta*qureg.stateVec[partner].real;
                qureg.stateVec[thisTask].imag = qureg.stateVec[thisTask].imag
                    + delta*qureg.stateVec[partner].imag;

                qureg.stateVec[partner].real = qureg.stateVec[partner].real + delta*real00;
                qureg.stateVec[partner].imag = qureg.stateVec[partner].imag + delta*imag00;

            }
        }

# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        //--------------------------------------- STEP THREE ---------------------
        for (thisTask=0; thisTask<numTasks; thisTask++){
            thisPatternQubit1 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit1;
            thisPatternQubit2 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit2;
            if ((thisPatternQubit2==0) && ((thisPatternQubit1==0)
                        || (thisPatternQubit1==totMaskQubit1))){
                //this element of form |...0...X...><...0...X...|  for X either 0 or 1.
                partner = thisTask | totMaskQubit2;
                partner = partner ^ totMaskQubit1;
                real00 =  qureg.stateVec[thisTask].real;
                imag00 =  qureg.stateVec[thisTask].imag;

                qureg.stateVec[thisTask].real = gamma * (qureg.stateVec[thisTask].real
                        + delta*qureg.stateVec[partner].real);
                qureg.stateVec[thisTask].imag = gamma * (qureg.stateVec[thisTask].imag
                        + delta*qureg.stateVec[partner].imag);

                qureg.stateVec[partner].real = gamma * (qureg.stateVec[partner].real
                        + delta*real00);
                qureg.stateVec[partner].imag = gamma * (qureg.stateVec[partner].imag
                        + delta*imag00);

            }
        }
    }
}

void densmatr_twoQubitDepolariseLocalPart1(Qureg qureg, int qubit1, int qubit2, qreal delta) {
    const long long int numTasks = qureg.numAmpsPerChunk;
    long long int innerMaskQubit1 = 1LL << qubit1;
    long long int outerMaskQubit1= 1LL << (qubit1 + qureg.numQubitsRepresented);
    long long int totMaskQubit1 = innerMaskQubit1 | outerMaskQubit1;
    long long int innerMaskQubit2 = 1LL << qubit2;
    long long int outerMaskQubit2 = 1LL << (qubit2 + qureg.numQubitsRepresented);
    long long int totMaskQubit2 = innerMaskQubit2 | outerMaskQubit2;
    // correct for being in a particular chunk
    //totMaskQubit2 = totMaskQubit2&(qureg.numAmpsPerChunk-1); // totMaskQubit2 % numAmpsPerChunk


    long long int thisTask;
    long long int partner;
    long long int thisPatternQubit1, thisPatternQubit2;

    qreal real00, imag00;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (totMaskQubit1,totMaskQubit2,qureg,delta) \
    private  (thisTask,partner,thisPatternQubit1,thisPatternQubit2,real00,imag00)
# endif
    {

# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        //--------------------------------------- STEP ONE ---------------------
        for (thisTask=0; thisTask<numTasks; thisTask ++){
            thisPatternQubit1 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit1;
            thisPatternQubit2 = (thisTask+qureg.numAmpsPerChunk*qureg.chunkId)&totMaskQubit2;
            if ((thisPatternQubit1==0) && ((thisPatternQubit2==0)
                        || (thisPatternQubit2==totMaskQubit2))){
                //this element of form |...X...0...><...X...0...|  for X either 0 or 1.
                partner = thisTask | totMaskQubit1;
                real00 =  qureg.stateVec[thisTask].real;
                imag00 =  qureg.stateVec[thisTask].imag;

                qureg.stateVec[thisTask].real = qureg.stateVec[thisTask].real
                    + delta*qureg.stateVec[partner].real;
                qureg.stateVec[thisTask].imag = qureg.stateVec[thisTask].imag
                    + delta*qureg.stateVec[partner].imag;

                qureg.stateVec[partner].real = qureg.stateVec[partner].real + delta*real00;
                qureg.stateVec[partner].imag = qureg.stateVec[partner].imag + delta*imag00;

            }
        }
    }
}

void densmatr_twoQubitDepolariseDistributed(Qureg qureg, const int targetQubit,
        const int qubit2, qreal delta, qreal gamma) {

    long long int sizeInnerBlockQ1, sizeInnerHalfBlockQ1;
    long long int sizeInnerBlockQ2, sizeInnerHalfBlockQ2, sizeInnerQuarterBlockQ2;
    long long int sizeOuterColumn, sizeOuterQuarterColumn;
    long long int thisInnerBlockQ2,
         thisOuterColumn, // current column in density matrix
         thisIndex,    // current index in (density matrix representation) state vector
         thisIndexInOuterColumn,
         thisIndexInInnerBlockQ1,
         thisIndexInInnerBlockQ2,
         thisInnerBlockQ1InInnerBlockQ2;
    int outerBitQ1, outerBitQ2;

    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>2;

    // set dimensions
    sizeInnerHalfBlockQ1 = 1LL << targetQubit;
    sizeInnerHalfBlockQ2 = 1LL << qubit2;
    sizeInnerQuarterBlockQ2 = sizeInnerHalfBlockQ2 >> 1;
    sizeInnerBlockQ2 = sizeInnerHalfBlockQ2 << 1;
    sizeInnerBlockQ1 = 2LL * sizeInnerHalfBlockQ1;
    sizeOuterColumn = 1LL << qureg.numQubitsRepresented;
    sizeOuterQuarterColumn = sizeOuterColumn >> 2;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (sizeInnerBlockQ1,sizeInnerHalfBlockQ1,sizeInnerBlockQ2,sizeInnerHalfBlockQ2,sizeInnerQuarterBlockQ2,\
                sizeOuterColumn,sizeOuterQuarterColumn,qureg,delta,gamma) \
    private  (thisTask,thisInnerBlockQ2,thisInnerBlockQ1InInnerBlockQ2, \
                thisOuterColumn,thisIndex,thisIndexInOuterColumn, \
                thisIndexInInnerBlockQ1,thisIndexInInnerBlockQ2,outerBitQ1,outerBitQ2)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        // thisTask iterates over half the elements in this process' chunk of the density matrix
        // treat this as iterating over all columns, then iterating over half the values
        // within one column.
        // If this function has been called, this process' chunk contains half an
        // outer block or less
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            // we want to process all columns in the density matrix,
            // updating the values for half of each column (one half of each inner block)
            thisOuterColumn = thisTask / sizeOuterQuarterColumn;
            // thisTask % sizeOuterQuarterColumn
            thisIndexInOuterColumn = thisTask&(sizeOuterQuarterColumn-1);
            thisInnerBlockQ2 = thisIndexInOuterColumn / sizeInnerQuarterBlockQ2;
            // thisTask % sizeInnerQuarterBlockQ2;
            thisIndexInInnerBlockQ2 = thisTask&(sizeInnerQuarterBlockQ2-1);
            thisInnerBlockQ1InInnerBlockQ2 = thisIndexInInnerBlockQ2 / sizeInnerHalfBlockQ1;
            // thisTask % sizeInnerHalfBlockQ1;
            thisIndexInInnerBlockQ1 = thisTask&(sizeInnerHalfBlockQ1-1);

            // get index in state vector corresponding to upper inner block
            thisIndex = thisOuterColumn*sizeOuterColumn + thisInnerBlockQ2*sizeInnerBlockQ2
                + thisInnerBlockQ1InInnerBlockQ2*sizeInnerBlockQ1 + thisIndexInInnerBlockQ1;

            // check if we are in the upper or lower half of an outer block for Q1
            outerBitQ1 = extractBit(targetQubit, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId)>>qureg.numQubitsRepresented);
            // if we are in the lower half of an outer block, shift to be in the lower half
            // of the inner block as well (we want to dephase |0><0| and |1><1| only)
            thisIndex += outerBitQ1*(sizeInnerHalfBlockQ1);

            // check if we are in the upper or lower half of an outer block for Q2
            outerBitQ2 = extractBit(qubit2, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId)>>qureg.numQubitsRepresented);
            // if we are in the lower half of an outer block, shift to be in the lower half
            // of the inner block as well (we want to dephase |0><0| and |1><1| only)
            thisIndex += outerBitQ2*(sizeInnerQuarterBlockQ2<<1);

            // NOTE: at this point thisIndex should be the index of the element we want to
            // dephase in the chunk of the state vector on this process, in the
            // density matrix representation.
            // thisTask is the index of the pair element in pairStateVec


            // state[thisIndex] = (1-depolLevel)*state[thisIndex] + depolLevel*(state[thisIndex]
            //      + pair[thisTask])/2
            // NOTE: must set gamma=1 if using this function for steps 1 or 2
            qureg.stateVec[thisIndex].real = gamma*(qureg.stateVec[thisIndex].real +
                    delta*qureg.pairStateVec[thisTask].real);
            qureg.stateVec[thisIndex].imag = gamma*(qureg.stateVec[thisIndex].imag +
                    delta*qureg.pairStateVec[thisTask].imag);
        }
    }
}

void densmatr_twoQubitDepolariseQ1LocalQ2DistributedPart3(Qureg qureg, const int targetQubit,
        const int qubit2, qreal delta, qreal gamma) {

    long long int sizeInnerBlockQ1, sizeInnerHalfBlockQ1;
    long long int sizeInnerBlockQ2, sizeInnerHalfBlockQ2, sizeInnerQuarterBlockQ2;
    long long int sizeOuterColumn, sizeOuterQuarterColumn;
    long long int thisInnerBlockQ2,
         thisOuterColumn, // current column in density matrix
         thisIndex,    // current index in (density matrix representation) state vector
         thisIndexInPairVector,
         thisIndexInOuterColumn,
         thisIndexInInnerBlockQ1,
         thisIndexInInnerBlockQ2,
         thisInnerBlockQ1InInnerBlockQ2;
    int outerBitQ1, outerBitQ2;

    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>2;

    // set dimensions
    sizeInnerHalfBlockQ1 = 1LL << targetQubit;
    sizeInnerHalfBlockQ2 = 1LL << qubit2;
    sizeInnerQuarterBlockQ2 = sizeInnerHalfBlockQ2 >> 1;
    sizeInnerBlockQ2 = sizeInnerHalfBlockQ2 << 1;
    sizeInnerBlockQ1 = 2LL * sizeInnerHalfBlockQ1;
    sizeOuterColumn = 1LL << qureg.numQubitsRepresented;
    sizeOuterQuarterColumn = sizeOuterColumn >> 2;

//# if 0
# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (sizeInnerBlockQ1,sizeInnerHalfBlockQ1,sizeInnerBlockQ2,sizeInnerHalfBlockQ2,sizeInnerQuarterBlockQ2,\
                sizeOuterColumn,sizeOuterQuarterColumn,qureg,delta,gamma) \
    private  (thisTask,thisInnerBlockQ2,thisInnerBlockQ1InInnerBlockQ2, \
                thisOuterColumn,thisIndex,thisIndexInPairVector,thisIndexInOuterColumn, \
                thisIndexInInnerBlockQ1,thisIndexInInnerBlockQ2,outerBitQ1,outerBitQ2)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
//# endif
        // thisTask iterates over half the elements in this process' chunk of the density matrix
        // treat this as iterating over all columns, then iterating over half the values
        // within one column.
        // If this function has been called, this process' chunk contains half an
        // outer block or less
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            // we want to process all columns in the density matrix,
            // updating the values for half of each column (one half of each inner block)
            thisOuterColumn = thisTask / sizeOuterQuarterColumn;
            // thisTask % sizeOuterQuarterColumn
            thisIndexInOuterColumn = thisTask&(sizeOuterQuarterColumn-1);
            thisInnerBlockQ2 = thisIndexInOuterColumn / sizeInnerQuarterBlockQ2;
            // thisTask % sizeInnerQuarterBlockQ2;
            thisIndexInInnerBlockQ2 = thisTask&(sizeInnerQuarterBlockQ2-1);
            thisInnerBlockQ1InInnerBlockQ2 = thisIndexInInnerBlockQ2 / sizeInnerHalfBlockQ1;
            // thisTask % sizeInnerHalfBlockQ1;
            thisIndexInInnerBlockQ1 = thisTask&(sizeInnerHalfBlockQ1-1);

            // get index in state vector corresponding to upper inner block
            thisIndex = thisOuterColumn*sizeOuterColumn + thisInnerBlockQ2*sizeInnerBlockQ2
                + thisInnerBlockQ1InInnerBlockQ2*sizeInnerBlockQ1 + thisIndexInInnerBlockQ1;

            // check if we are in the upper or lower half of an outer block for Q1
            outerBitQ1 = extractBit(targetQubit, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId)>>qureg.numQubitsRepresented);
            // if we are in the lower half of an outer block, shift to be in the lower half
            // of the inner block as well (we want to dephase |0><0| and |1><1| only)
            thisIndex += outerBitQ1*(sizeInnerHalfBlockQ1);

            // For part 3 we need to match elements such that (my Q1 != pair Q1) AND (my Q2 != pair Q2)
            // Find correct index in pairStateVector
            thisIndexInPairVector = thisTask + (1-outerBitQ1)*sizeInnerHalfBlockQ1*sizeOuterQuarterColumn -
                outerBitQ1*sizeInnerHalfBlockQ1*sizeOuterQuarterColumn;

            // check if we are in the upper or lower half of an outer block for Q2
            outerBitQ2 = extractBit(qubit2, (thisIndex+qureg.numAmpsPerChunk*qureg.chunkId)>>qureg.numQubitsRepresented);
            // if we are in the lower half of an outer block, shift to be in the lower half
            // of the inner block as well (we want to dephase |0><0| and |1><1| only)
            thisIndex += outerBitQ2*(sizeInnerQuarterBlockQ2<<1);


            // NOTE: at this point thisIndex should be the index of the element we want to
            // dephase in the chunk of the state vector on this process, in the
            // density matrix representation.


            // state[thisIndex] = (1-depolLevel)*state[thisIndex] + depolLevel*(state[thisIndex]
            //      + pair[thisIndexInPairVector])/2
            qureg.stateVec[thisIndex].real = gamma*(qureg.stateVec[thisIndex].real +
                    delta*qureg.pairStateVec[thisIndexInPairVector].real);

            qureg.stateVec[thisIndex].imag = gamma*(qureg.stateVec[thisIndex].imag +
                    delta*qureg.pairStateVec[thisIndexInPairVector].imag);
        }
    }

}


/* Without nested parallelisation, only the outer most loops which call below are parallelised */
void zeroSomeAmps(Qureg qureg, long long int startInd, long long int numAmps) {

# ifdef _OPENMP
# pragma omp parallel for schedule (static)
# endif
    for (long long int i=startInd; i < startInd+numAmps; i++) {
        qureg.stateVec[i].real = 0;
        qureg.stateVec[i].imag = 0;
    }
}
void normaliseSomeAmps(Qureg qureg, qreal norm, long long int startInd, long long int numAmps) {

# ifdef _OPENMP
# pragma omp parallel for schedule (static)
# endif
    for (long long int i=startInd; i < startInd+numAmps; i++) {
        qureg.stateVec[i].real /= norm;
        qureg.stateVec[i].imag /= norm;
    }
}
void alternateNormZeroingSomeAmpBlocks(
    Qureg qureg, qreal norm, int normFirst,
    long long int startAmpInd, long long int numAmps, long long int blockSize
) {
    long long int numDubBlocks = numAmps / (2*blockSize);
    long long int blockStartInd;

    if (normFirst) {

# ifdef _OPENMP
# pragma omp parallel for schedule (static) private (blockStartInd)
# endif
        for (long long int dubBlockInd=0; dubBlockInd < numDubBlocks; dubBlockInd++) {
            blockStartInd = startAmpInd + dubBlockInd*2*blockSize;
            normaliseSomeAmps(qureg, norm, blockStartInd,             blockSize); // |0><0|
            zeroSomeAmps(     qureg,       blockStartInd + blockSize, blockSize);
        }
    } else {

# ifdef _OPENMP
# pragma omp parallel for schedule (static) private (blockStartInd)
# endif
        for (long long int dubBlockInd=0; dubBlockInd < numDubBlocks; dubBlockInd++) {
            blockStartInd = startAmpInd + dubBlockInd*2*blockSize;
            zeroSomeAmps(     qureg,       blockStartInd,             blockSize);
            normaliseSomeAmps(qureg, norm, blockStartInd + blockSize, blockSize); // |1><1|
        }
    }
}

/** Renorms (/prob) every | * outcome * >< * outcome * | state, setting all others to zero */
void densmatr_collapseToKnownProbOutcome(Qureg qureg, const int measureQubit, int outcome, qreal totalStateProb) {

	// only (global) indices (as bit sequence): '* outcome *(n+q) outcome *q are spared
    // where n = measureQubit, q = qureg.numQubitsRepresented.
    // We can thus step in blocks of 2^q+n, killing every second, and inside the others,
    //  stepping in sub-blocks of 2^q, killing every second.
    // When outcome=1, we offset the start of these blocks by their size.
    long long int innerBlockSize = (1LL << measureQubit);
    long long int outerBlockSize = (1LL << (measureQubit + qureg.numQubitsRepresented));

    // Because there are 2^a number of nodes(/chunks), each node will contain 2^b number of blocks,
    // or each block will span 2^c number of nodes. Similarly for the innerblocks.
    long long int locNumAmps = qureg.numAmpsPerChunk;
    long long int globalStartInd = qureg.chunkId * locNumAmps;
    int innerBit = extractBit(measureQubit, globalStartInd);
    int outerBit = extractBit(measureQubit + qureg.numQubitsRepresented, globalStartInd);

    // If this chunk's amps are entirely inside an outer block
    if (locNumAmps <= outerBlockSize) {

        // if this is an undesired outer block, kill all elems
        if (outerBit != outcome)
            return zeroSomeAmps(qureg, 0, qureg.numAmpsPerChunk);

        // othwerwise, if this is a desired outer block, and also entirely an inner block
        if (locNumAmps <= innerBlockSize) {

            // and that inner block is undesired, kill all elems
            if (innerBit != outcome)
                return zeroSomeAmps(qureg, 0, qureg.numAmpsPerChunk);
            // otherwise normalise all elems
            else
                return normaliseSomeAmps(qureg, totalStateProb, 0, qureg.numAmpsPerChunk);
        }

        // otherwise this is a desired outer block which contains 2^a inner blocks; kill/renorm every second inner block
        return alternateNormZeroingSomeAmpBlocks(
            qureg, totalStateProb, innerBit==outcome, 0, qureg.numAmpsPerChunk, innerBlockSize);
    }

    // Otherwise, this chunk's amps contain multiple outer blocks (and hence multiple inner blocks)
    long long int numOuterDoubleBlocks = locNumAmps / (2*outerBlockSize);
    long long int firstBlockInd;

    // alternate norming* and zeroing the outer blocks (with order based on the desired outcome)
    // These loops aren't parallelised, since they could have 1 or 2 iterations and will prevent
    // inner parallelisation
    if (outerBit == outcome) {

        for (long long int outerDubBlockInd = 0; outerDubBlockInd < numOuterDoubleBlocks; outerDubBlockInd++) {
            firstBlockInd = outerDubBlockInd*2*outerBlockSize;

            // *norm only the desired inner blocks in the desired outer block
            alternateNormZeroingSomeAmpBlocks(
                qureg, totalStateProb, innerBit==outcome,
                firstBlockInd, outerBlockSize, innerBlockSize);

            // zero the undesired outer block
            zeroSomeAmps(qureg, firstBlockInd + outerBlockSize, outerBlockSize);
        }

    } else {

        for (long long int outerDubBlockInd = 0; outerDubBlockInd < numOuterDoubleBlocks; outerDubBlockInd++) {
            firstBlockInd = outerDubBlockInd*2*outerBlockSize;

            // same thing but undesired outer blocks come first
            zeroSomeAmps(qureg, firstBlockInd, outerBlockSize);
            alternateNormZeroingSomeAmpBlocks(
                qureg, totalStateProb, innerBit==outcome,
                firstBlockInd + outerBlockSize, outerBlockSize, innerBlockSize);
        }
    }

}

qreal densmatr_calcPurityLocal(Qureg qureg) {

    /* sum of qureg^2, which is sum_i |qureg[i]|^2 */
    long long int index;
    long long int numAmps = qureg.numAmpsPerChunk;

    qreal trace = 0;

# ifdef _OPENMP
# pragma omp parallel \
    shared    (qureg, numAmps) \
    private   (index) \
    reduction ( +:trace )
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
        for (index=0LL; index<numAmps; index++) {

            trace += qureg.stateVec[index].real*qureg.stateVec[index].real +
                qureg.stateVec[index].imag*qureg.stateVec[index].imag;
        }
    }

    return trace;
}

void densmatr_addDensityMatrix(Qureg combineQureg, qreal otherProb, Qureg otherQureg) {

    /* corresponding amplitudes live on the same node (same dimensions) */
    long long int numAmps = combineQureg.numAmpsPerChunk;
    long long int index;

# ifdef _OPENMP
# pragma omp parallel \
    default (none) \
    shared  (combineQureg, otherQureg, otherProb, numAmps) \
    private (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
        for (index=0; index < numAmps; index++) {
            combineQureg.stateVec[index].real *= 1-otherProb;
            combineQureg.stateVec[index].imag *= 1-otherProb;

            combineQureg.stateVec[index].real += otherProb * otherQureg.stateVec[index].real;
            combineQureg.stateVec[index].imag += otherProb * otherQureg.stateVec[index].imag;
        }
    }
}

/** computes a few dens-columns-worth of (vec^*T) dens * vec */
qreal densmatr_calcFidelityLocal(Qureg qureg, Qureg pureState) {

    /* Here, elements of pureState are not accessed (instead grabbed from qureg.pair).
     * We only consult the attributes.
     *
     * qureg is a density matrix, and pureState is a statevector.
     * Every node contains as many columns of qureg as amps by pureState.
     * Ergo, this node contains columns:
     * qureg.chunkID * pureState.numAmpsPerChunk  to
     * (qureg.chunkID + 1) * pureState.numAmpsPerChunk
     *
     * The first pureState.numAmpsTotal elements of qureg.pairStateVec are the
     * full pure state-vector
     */

    int row, col;
    int dim = pureState.numAmpsTotal;
    int colsPerNode = pureState.numAmpsPerChunk;

    qreal densElemRe, densElemIm;
    qreal prefacRe, prefacIm;
    qreal rowSumRe, rowSumIm;
    qreal vecElemRe, vecElemIm;

    // starting GLOBAL column index of the qureg columns on this node
    int startCol = qureg.chunkId * pureState.numAmpsPerChunk;

    // quantity computed by this node
    qreal globalSumRe = 0;   // imag-component is assumed zero

# ifdef _OPENMP
# pragma omp parallel \
    shared    (qureg, dim,colsPerNode,startCol) \
    private   (row,col, prefacRe,prefacIm, rowSumRe,rowSumIm, densElemRe,densElemIm, vecElemRe,vecElemIm) \
    reduction ( +:globalSumRe )
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
        // indices of my GLOBAL row
        for (row=0; row < dim; row++) {

            // single element of conj(pureState)
            prefacRe =   qureg.pairStateVec[row].real;
            prefacIm = - qureg.pairStateVec[row].imag;

            rowSumRe = 0;
            rowSumIm = 0;

            // indices of my LOCAL column
            for (col=0; col < colsPerNode; col++) {

                // my local density element
                densElemRe = qureg.stateVec[row + dim*col].real;
                densElemIm = qureg.stateVec[row + dim*col].imag;

                // state-vector element
                vecElemRe = qureg.pairStateVec[startCol + col].real;
                vecElemIm = qureg.pairStateVec[startCol + col].imag;

                rowSumRe += densElemRe*vecElemRe - densElemIm*vecElemIm;
                rowSumIm += densElemRe*vecElemIm + densElemIm*vecElemRe;
            }

            globalSumRe += rowSumRe*prefacRe - rowSumIm*prefacIm;
        }
    }

    return globalSumRe;
}

Complex statevec_calcInnerProductLocal(Qureg bra, Qureg ket) {

    qreal innerProdReal = 0;
    qreal innerProdImag = 0;

    long long int index;
    long long int numAmps = bra.numAmpsPerChunk;

    qreal braRe, braIm, ketRe, ketIm;

# ifdef _OPENMP
# pragma omp parallel \
    shared    (bra, ket, numAmps)    \
    private   (index, braRe, braIm, ketRe, ketIm) \
    reduction ( +:innerProdReal, innerProdImag )
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
        for (index=0; index < numAmps; index++) {
            braRe = bra.stateVec[index].real;
            braIm = bra.stateVec[index].imag;
            ketRe = ket.stateVec[index].real;
            ketIm = ket.stateVec[index].imag;

            // conj(bra_i) * ket_i
            innerProdReal += braRe*ketRe + braIm*ketIm;
            innerProdImag += braRe*ketIm - braIm*ketRe;
        }
    }

    Complex innerProd;
    innerProd.real = innerProdReal;
    innerProd.imag = innerProdImag;
    return innerProd;
}



void densmatr_initClassicalState (Qureg qureg, long long int stateInd)
{
    // dimension of the state vector
    long long int densityNumElems = qureg.numAmpsPerChunk;

    // initialise the state to all zeros
    long long int index;
# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, densityNumElems)  \
    private  (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<densityNumElems; index++) {
            qureg.stateVec[index].real = 0.0;
            qureg.stateVec[index].imag = 0.0;
        }
    }

    // index of the single density matrix elem to set non-zero
    long long int densityDim = 1LL << qureg.numQubitsRepresented;
    long long int densityInd = (densityDim + 1)*stateInd;

    // give the specified classical state prob 1
    if (qureg.chunkId == densityInd / densityNumElems){
        qureg.stateVec[densityInd % densityNumElems].real = 1.0;
        qureg.stateVec[densityInd % densityNumElems].imag = 0.0;
    }
}


void densmatr_initPlusState (Qureg qureg)
{
    // |+><+| = sum_i 1/sqrt(2^N) |i> 1/sqrt(2^N) <j| = sum_ij 1/2^N |i><j|
    long long int dim = (1LL << qureg.numQubitsRepresented);
    qreal probFactor = 1.0/((qreal) dim);

    long long int index;
    long long int chunkSize = qureg.numAmpsPerChunk;
    // initialise the state to |+++..+++> = 1/normFactor {1, 1, 1, ...}
# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, chunkSize, probFactor)      \
    private  (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<chunkSize; index++) {
            qureg.stateVec[index].real = probFactor;
            qureg.stateVec[index].imag = 0.0;
        }
    }
}

void densmatr_initPureStateLocal(Qureg targetQureg, Qureg copyQureg) {

    /* copyQureg amps aren't explicitly used - they're accessed through targetQureg.pair,
     * which contains the full pure statevector.
     * targetQureg has as many columns on node as copyQureg has amps
     */

    long long int colOffset = targetQureg.chunkId * copyQureg.numAmpsPerChunk;
    long long int colsPerNode = copyQureg.numAmpsPerChunk;
    long long int rowsPerNode = copyQureg.numAmpsTotal;

    long long int col, row, index;

    // a_i conj(a_j) |i><j|
    qreal ketRe, ketIm, braRe, braIm;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (targetQureg, colOffset, colsPerNode,rowsPerNode)   \
    private  (col,row, ketRe,ketIm,braRe,braIm, index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        // local column
        for (col=0; col < colsPerNode; col++) {

            // global row
            for (row=0; row < rowsPerNode; row++) {

                // get pure state amps
                ketRe = targetQureg.pairStateVec[row].real;
                ketIm = targetQureg.pairStateVec[row].imag;
                braRe = targetQureg.pairStateVec[col + colOffset].real;
                braIm = targetQureg.pairStateVec[col + colOffset].imag;

                // update density matrix
                index = row + col*rowsPerNode; // local ind
                targetQureg.stateVec[index].real = ketRe*braRe - ketIm*braIm;
                targetQureg.stateVec[index].imag = ketRe*braIm - ketIm*braRe;
            }
        }
    }
}

void statevec_setAmps(Qureg qureg, long long int startInd, qreal* reals, qreal* imags, long long int numAmps) {

    /* this is actually distributed, since the user's code runs on every node */

    // local start/end indices of the given amplitudes, assuming they fit in this chunk
    // these may be negative or above qureg.numAmpsPerChunk
    long long int localStartInd = startInd - qureg.chunkId*qureg.numAmpsPerChunk;
    long long int localEndInd = localStartInd + numAmps; // exclusive

    // add this to a local index to get corresponding elem in reals & imags
    long long int offset = qureg.chunkId*qureg.numAmpsPerChunk - startInd;

    // restrict these indices to fit into this chunk
    if (localStartInd < 0)
        localStartInd = 0;
    if (localEndInd > qureg.numAmpsPerChunk)
        localEndInd = qureg.numAmpsPerChunk;
    // they may now be out of order = no iterations

    // unpacking OpenMP vars
    long long int index;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, localStartInd,localEndInd, reals,imags, offset) \
    private  (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        // iterate these local inds - this might involve no iterations
        for (index=localStartInd; index < localEndInd; index++) {
            qureg.stateVec[index].real = reals[index + offset];
            qureg.stateVec[index].imag = imags[index + offset];
        }
    }
}

void statevec_createQureg(Qureg *qureg, int numQubits, QuESTEnv env)
{
    long long int numAmps = 1L << numQubits;
    long long int numAmpsPerRank = numAmps/env.numRanks;

    qureg->stateVec = malloc(numAmpsPerRank * sizeof(*(qureg->stateVec)));
    if (env.numRanks>1){
        qureg->pairStateVec = malloc(numAmpsPerRank * sizeof(*(qureg->pairStateVec)));
    }

    if ( (!(qureg->stateVec) && numAmpsPerRank ) ) {
        printf("Could not allocate memory!");
        exit (EXIT_FAILURE);
    }

    if ( env.numRanks>1 && (!(qureg->pairStateVec) && numAmpsPerRank ) ) {
        printf("Could not allocate memory!");
        exit (EXIT_FAILURE);
    }

    qureg->numQubitsInStateVec = numQubits;
    qureg->numAmpsTotal = numAmps;
    qureg->numAmpsPerChunk = numAmpsPerRank;
    qureg->chunkId = env.rank;
    qureg->numChunks = env.numRanks;
    qureg->isDensityMatrix = 0;
}

void statevec_destroyQureg(Qureg qureg, QuESTEnv env){

    qureg.numQubitsInStateVec = 0;
    qureg.numAmpsTotal = 0;
    qureg.numAmpsPerChunk = 0;

    free(qureg.stateVec);
    if (env.numRanks>1){
        free(qureg.pairStateVec);
    }
    qureg.stateVec = NULL;
    qureg.pairStateVec = NULL;
}

void statevec_reportStateToScreen(Qureg qureg, QuESTEnv env, int reportRank){
    long long int index;
    int rank;
    if (qureg.numQubitsInStateVec<=5){
        for (rank=0; rank<qureg.numChunks; rank++){
            if (qureg.chunkId==rank){
                if (reportRank) {
                    printf("Reporting state from rank %d [\n", qureg.chunkId);
                    printf("real, imag\n");
                } else if (rank==0) {
                    printf("Reporting state [\n");
                    printf("real, imag\n");
                }

                for(index=0; index<qureg.numAmpsPerChunk; index++){
                    //printf(REAL_STRING_FORMAT ", " REAL_STRING_FORMAT "\n", qureg.pairStateVec[index].real, qureg.pairStateVec[index].imag);
                    printf(REAL_STRING_FORMAT ", " REAL_STRING_FORMAT "\n", qureg.stateVec[index].real, qureg.stateVec[index].imag);
                }
                if (reportRank || rank==qureg.numChunks-1) printf("]\n");
            }
            syncQuESTEnv(env);
        }
    } else printf("Error: reportStateToScreen will not print output for systems of more than 5 qubits.\n");
}
void statevec_getEnvironmentString(QuESTEnv env, Qureg qureg, char str[200]){
    int numThreads=1;
# ifdef _OPENMP
    numThreads=omp_get_max_threads();
# endif
    sprintf(str, "%dqubits_CPU_%dranksx%dthreads", qureg.numQubitsInStateVec, env.numRanks, numThreads);
}

void statevec_initZeroState (Qureg qureg)
{
    long long int stateVecSize;
    long long int index;

    // dimension of the state vector
    stateVecSize = qureg.numAmpsPerChunk;

    // Can't use qureg->stateVec as a private OMP var

    // initialise the state-vector to all-zeroes
# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, stateVecSize) \
    private  (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<stateVecSize; index++) {
            qureg.stateVec[index].real = 0.0;
            qureg.stateVec[index].imag = 0.0;
        }
    }

    if (qureg.chunkId==0){
        // zero state |0000..0000> has probability 1
        qureg.stateVec[0].real = 0.0;
        qureg.stateVec[0].imag = 0.0;
    }
}

void statevec_initPlusState (Qureg qureg)
{
    long long int chunkSize, stateVecSize;
    long long int index;

    // dimension of the state vector
    chunkSize = qureg.numAmpsPerChunk;
    stateVecSize = chunkSize*qureg.numChunks;
    qreal normFactor = 1.0/sqrt((qreal)stateVecSize);

    // Can't use qureg->stateVec as a private OMP var

    // initialise the state to |+++..+++> = 1/normFactor {1, 1, 1, ...}
# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, chunkSize, normFactor) \
    private  (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<chunkSize; index++) {
            qureg.stateVec[index].real = normFactor;
            qureg.stateVec[index].imag = 0.0;
        }
    }
}

void statevec_initClassicalState (Qureg qureg, long long int stateInd)
{
    long long int stateVecSize;
    long long int index;

    // dimension of the state vector
    stateVecSize = qureg.numAmpsPerChunk;

    // Can't use qureg->stateVec as a private OMP var

    // initialise the state to vector to all zeros
# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, stateVecSize) \
    private  (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<stateVecSize; index++) {
            qureg.stateVec[index].real = 0.0;
            qureg.stateVec[index].imag = 0.0;
        }
    }

    // give the specified classical state prob 1
    if (qureg.chunkId == stateInd/stateVecSize){
        qureg.stateVec[stateInd % stateVecSize].real = 1.0;
        qureg.stateVec[stateInd % stateVecSize].imag = 0.0;
    }
}

void statevec_cloneQureg(Qureg targetQureg, Qureg copyQureg) {

    // registers are equal sized, so nodes hold the same state-vector partitions
    long long int stateVecSize;
    long long int index;

    // dimension of the state vector
    stateVecSize = targetQureg.numAmpsPerChunk;

    // initialise the state to |0000..0000>
# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (targetQureg, copyQureg, stateVecSize) \
    private  (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<stateVecSize; index++) {
            targetQureg.stateVec[index].real = copyQureg.stateVec[index].real;
            targetQureg.stateVec[index].imag = copyQureg.stateVec[index].imag;
        }
    }
}

/**
 * Initialise the state vector of probability amplitudes such that one qubit is set to 'outcome' and all other qubits are in an equal superposition of zero and one.
 * @param[in,out] qureg object representing the set of qubits to be initialised
 * @param[in] qubitId id of qubit to set to state 'outcome'
 * @param[in] value of qubit 'qubitId'
 */
void statevec_initStateOfSingleQubit(Qureg *qureg, int qubitId, int outcome)
{
    long long int chunkSize, stateVecSize;
    long long int index;
    int bit;
    const long long int chunkId=qureg->chunkId;

    // dimension of the state vector
    chunkSize = qureg->numAmpsPerChunk;
    stateVecSize = chunkSize*qureg->numChunks;
    qreal normFactor = 1.0/sqrt((qreal)stateVecSize/2.0);

    // Can't use qureg->stateVec as a private OMP var

    // initialise the state to |0000..0000>
# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, chunkSize, normFactor, qubitId, outcome) \
    private  (index, bit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<chunkSize; index++) {
            bit = extractBit(qubitId, index+chunkId*chunkSize);
            if (bit==outcome) {
                qureg->stateVec[index].real = normFactor;
                qureg->stateVec[index].imag = 0.0;
            } else {
                qureg->stateVec[index].real = 0.0;
                qureg->stateVec[index].imag = 0.0;
            }
        }
    }
}


/**
 * Initialise the state vector of probability amplitudes to an (unphysical) state with
 * each component of each probability amplitude a unique floating point value. For debugging processes
 * @param[in,out] qureg object representing the set of qubits to be initialised
 */
void statevec_initStateDebug (Qureg qureg)
{
    long long int chunkSize;
    long long int index;
    long long int indexOffset;

    // dimension of the state vector
    chunkSize = qureg.numAmpsPerChunk;

    // Can't use qureg->stateVec as a private OMP var

    indexOffset = chunkSize * qureg.chunkId;

    // initialise the state to |0000..0000>
# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, chunkSize, indexOffset) \
    private  (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<chunkSize; index++) {
            qureg.stateVec[index].real = ((indexOffset + index)*2.0)/10.0;
            qureg.stateVec[index].imag = ((indexOffset + index)*2.0+1.0)/10.0;
        }
    }
}

// returns 1 if successful, else 0
int statevec_initStateFromSingleFile(Qureg *qureg, char filename[200], QuESTEnv env){
    long long int chunkSize, stateVecSize;
    long long int indexInChunk, totalIndex;

    chunkSize = qureg->numAmpsPerChunk;
    stateVecSize = chunkSize*qureg->numChunks;


    FILE *fp;
    char line[200];

    for (int rank=0; rank<(qureg->numChunks); rank++){
        if (rank==qureg->chunkId){
            fp = fopen(filename, "r");

            // indicate file open failure
            if (fp == NULL)
                return 0;

            indexInChunk = 0; totalIndex = 0;
            while (fgets(line, sizeof(char)*200, fp) != NULL && totalIndex<stateVecSize){
                if (line[0]!='#'){
                    int chunkId = totalIndex/chunkSize;
                    if (chunkId==qureg->chunkId){
                        # if QuEST_PREC==1
                        sscanf(line, "%f, %f", &(qureg->stateVec[indexInChunk].real),
                                &(qureg->stateVec[indexInChunk].imag));
                        # elif QuEST_PREC==2
                        sscanf(line, "%lf, %lf", &(qureg->stateVec[indexInChunk].real),
                                &(qureg->stateVec[indexInChunk].imag));
                        # elif QuEST_PREC==4
                        sscanf(line, "%Lf, %Lf", &(qureg->stateVec[indexInChunk].real),
                                &(qureg->stateVec[indexInChunk].imag));
                        # endif
                        indexInChunk += 1;
                    }
                    totalIndex += 1;
                }
            }
            fclose(fp);
        }
        syncQuESTEnv(env);
    }

    // indicate success
    return 1;
}

int statevec_compareStates(Qureg mq1, Qureg mq2, qreal precision){
    qreal diff;
    int chunkSize = mq1.numAmpsPerChunk;

    for (int i=0; i<chunkSize; i++){
        diff = absReal(mq1.stateVec[i].real - mq2.stateVec[i].real);
        if (diff>precision) return 0;
        diff = absReal(mq1.stateVec[i].imag - mq2.stateVec[i].imag);
        if (diff>precision) return 0;
    }
    return 1;
}

void statevec_compactUnitaryLocal (Qureg qureg, const int targetQubit, Complex alpha, Complex beta)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var
    qreal alphaImag=alpha.imag, alphaReal=alpha.real;
    qreal betaImag=beta.imag, betaReal=beta.real;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock, alphaReal,alphaImag, betaReal,betaImag) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp,stateRealLo,stateImagLo)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {

            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            // store current state vector values in temp variables
            stateRealUp = qureg.stateVec[indexUp].real;
            stateImagUp = qureg.stateVec[indexUp].imag;

            stateRealLo = qureg.stateVec[indexLo].real;
            stateImagLo = qureg.stateVec[indexLo].imag;

            // state[indexUp] = alpha * state[indexUp] - conj(beta)  * state[indexLo]
            qureg.stateVec[indexUp].real = alphaReal*stateRealUp - alphaImag*stateImagUp
                - betaReal*stateRealLo - betaImag*stateImagLo;
            qureg.stateVec[indexUp].imag = alphaReal*stateImagUp + alphaImag*stateRealUp
                - betaReal*stateImagLo + betaImag*stateRealLo;

            // state[indexLo] = beta  * state[indexUp] + conj(alpha) * state[indexLo]
            qureg.stateVec[indexLo].real = betaReal*stateRealUp - betaImag*stateImagUp
                + alphaReal*stateRealLo + alphaImag*stateImagLo;
            qureg.stateVec[indexLo].imag = betaReal*stateImagUp + betaImag*stateRealUp
                + alphaReal*stateImagLo - alphaImag*stateRealLo;
        }
    }

}

void statevec_unitaryLocal(Qureg qureg, const int targetQubit, ComplexMatrix2 u)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock, u) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp,stateRealLo,stateImagLo)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {

            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            // store current state vector values in temp variables
            stateRealUp = qureg.stateVec[indexUp].real;
            stateImagUp = qureg.stateVec[indexUp].imag;

            stateRealLo = qureg.stateVec[indexLo].real;
            stateImagLo = qureg.stateVec[indexLo].imag;


            // state[indexUp] = u00 * state[indexUp] + u01 * state[indexLo]
            qureg.stateVec[indexUp].real = u.r0c0.real*stateRealUp - u.r0c0.imag*stateImagUp
                + u.r0c1.real*stateRealLo - u.r0c1.imag*stateImagLo;
            qureg.stateVec[indexUp].imag = u.r0c0.real*stateImagUp + u.r0c0.imag*stateRealUp
                + u.r0c1.real*stateImagLo + u.r0c1.imag*stateRealLo;

            // state[indexLo] = u10  * state[indexUp] + u11 * state[indexLo]
            qureg.stateVec[indexLo].real = u.r1c0.real*stateRealUp  - u.r1c0.imag*stateImagUp
                + u.r1c1.real*stateRealLo  -  u.r1c1.imag*stateImagLo;
            qureg.stateVec[indexLo].imag = u.r1c0.real*stateImagUp + u.r1c0.imag*stateRealUp
                + u.r1c1.real*stateImagLo + u.r1c1.imag*stateRealLo;

        }
    }
}

/** Rotate a single qubit in the state vector of probability amplitudes,
 * given two complex numbers alpha and beta,
 * and a subset of the state vector with upper and lower block values stored seperately.
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] targetQubit qubit to rotate
 *  @param[in] rot1 rotation angle
 *  @param[in] rot2 rotation angle
 *  @param[in] stateVecUp probability amplitudes in upper half of a block
 *  @param[in] stateVecLo probability amplitudes in lower half of a block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_compactUnitaryDistributed (Qureg qureg, const int targetQubit,
        Complex rot1, Complex rot2,
        Complex *stateVecUp,
        Complex *stateVecLo,
        Complex *stateVecOut)
{

    qreal   stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;

    qreal rot1Real=rot1.real, rot1Imag=rot1.imag;
    qreal rot2Real=rot2.real, rot2Imag=rot2.imag;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (stateVecUp,stateVecLo,stateVecOut, \
            rot1Real,rot1Imag, rot2Real,rot2Imag) \
    private  (thisTask,stateRealUp,stateImagUp,stateRealLo,stateImagLo)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            // store current state vector values in temp variables
            stateRealUp = stateVecUp[thisTask].real;
            stateImagUp = stateVecUp[thisTask].imag;

            stateRealLo = stateVecLo[thisTask].real;
            stateImagLo = stateVecLo[thisTask].imag;

            // state[indexUp] = alpha * state[indexUp] - conj(beta)  * state[indexLo]
            stateVecOut[thisTask].real = rot1Real*stateRealUp - rot1Imag*stateImagUp + rot2Real*stateRealLo + rot2Imag*stateImagLo;
            stateVecOut[thisTask].imag = rot1Real*stateImagUp + rot1Imag*stateRealUp + rot2Real*stateImagLo - rot2Imag*stateRealLo;
        }
    }
}

/** Apply a unitary operation to a single qubit
 *  given a subset of the state vector with upper and lower block values
 * stored seperately.
 *
 *  @remarks Qubits are zero-based and the first qubit is the rightmost
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] targetQubit qubit to rotate
 *  @param[in] u unitary matrix to apply
 *  @param[in] stateVecUp probability amplitudes in upper half of a block
 *  @param[in] stateVecLo probability amplitudes in lower half of a block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_unitaryDistributed (Qureg qureg, const int targetQubit,
        Complex rot1, Complex rot2,
        Complex *stateVecUp,
        Complex *stateVecLo,
        Complex *stateVecOut)
{

    qreal   stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;

    qreal rot1Real=rot1.real, rot1Imag=rot1.imag;
    qreal rot2Real=rot2.real, rot2Imag=rot2.imag;


# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (stateVecUp,stateVecLo,stateVecOut, \
            rot1Real, rot1Imag, rot2Real, rot2Imag) \
    private  (thisTask,stateRealUp,stateImagUp,stateRealLo,stateImagLo)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            // store current state vector values in temp variables
            stateRealUp = stateVecUp[thisTask].real;
            stateImagUp = stateVecUp[thisTask].imag;

            stateRealLo = stateVecLo[thisTask].real;
            stateImagLo = stateVecLo[thisTask].imag;

            stateVecOut[thisTask].real = rot1Real*stateRealUp - rot1Imag*stateImagUp
                + rot2Real*stateRealLo - rot2Imag*stateImagLo;
            stateVecOut[thisTask].imag = rot1Real*stateImagUp + rot1Imag*stateRealUp
                + rot2Real*stateImagLo + rot2Imag*stateRealLo;
        }
    }
}

void statevec_controlledCompactUnitaryLocal (Qureg qureg, const int controlQubit, const int targetQubit,
        Complex alpha, Complex beta)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    int controlBit;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var
    qreal alphaImag=alpha.imag, alphaReal=alpha.real;
    qreal betaImag=beta.imag, betaReal=beta.real;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock, alphaReal,alphaImag, betaReal,betaImag) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp,stateRealLo,stateImagLo,controlBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {

            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            controlBit = extractBit (controlQubit, indexUp+chunkId*chunkSize);
            if (controlBit){
                // store current state vector values in temp variables
                stateRealUp = qureg.stateVec[indexUp].real;
                stateImagUp = qureg.stateVec[indexUp].imag;

                stateRealLo = qureg.stateVec[indexLo].real;
                stateImagLo = qureg.stateVec[indexLo].imag;

                // state[indexUp] = alpha * state[indexUp] - conj(beta)  * state[indexLo]
                qureg.stateVec[indexUp].real = alphaReal*stateRealUp - alphaImag*stateImagUp
                    - betaReal*stateRealLo - betaImag*stateImagLo;
                qureg.stateVec[indexUp].imag = alphaReal*stateImagUp + alphaImag*stateRealUp
                    - betaReal*stateImagLo + betaImag*stateRealLo;

                // state[indexLo] = beta  * state[indexUp] + conj(alpha) * state[indexLo]
                qureg.stateVec[indexLo].real = betaReal*stateRealUp - betaImag*stateImagUp
                    + alphaReal*stateRealLo + alphaImag*stateImagLo;
                qureg.stateVec[indexLo].imag = betaReal*stateImagUp + betaImag*stateRealUp
                    + alphaReal*stateImagLo - alphaImag*stateRealLo;
            }
        }
    }

}

void statevec_multiControlledUnitaryLocal(Qureg qureg, const int targetQubit,
        long long int mask, ComplexMatrix2 u)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock, u, mask) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp,stateRealLo,stateImagLo)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {

            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            if (mask == (mask & (indexUp+chunkId*chunkSize)) ){
                // store current state vector values in temp variables
                stateRealUp = qureg.stateVec[indexUp].real;
                stateImagUp = qureg.stateVec[indexUp].imag;

                stateRealLo = qureg.stateVec[indexLo].real;
                stateImagLo = qureg.stateVec[indexLo].imag;


                // state[indexUp] = u00 * state[indexUp] + u01 * state[indexLo]
                qureg.stateVec[indexUp].real = u.r0c0.real*stateRealUp - u.r0c0.imag*stateImagUp
                    + u.r0c1.real*stateRealLo - u.r0c1.imag*stateImagLo;
                qureg.stateVec[indexUp].imag = u.r0c0.real*stateImagUp + u.r0c0.imag*stateRealUp
                    + u.r0c1.real*stateImagLo + u.r0c1.imag*stateRealLo;

                // state[indexLo] = u10  * state[indexUp] + u11 * state[indexLo]
                qureg.stateVec[indexLo].real = u.r1c0.real*stateRealUp  - u.r1c0.imag*stateImagUp
                    + u.r1c1.real*stateRealLo  -  u.r1c1.imag*stateImagLo;
                qureg.stateVec[indexLo].imag = u.r1c0.real*stateImagUp + u.r1c0.imag*stateRealUp
                    + u.r1c1.real*stateImagLo + u.r1c1.imag*stateRealLo;
            }
        }
    }

}

void statevec_controlledUnitaryLocal(Qureg qureg, const int controlQubit, const int targetQubit,
        ComplexMatrix2 u)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    int controlBit;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock, u) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp,stateRealLo,stateImagLo,controlBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {

            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            controlBit = extractBit (controlQubit, indexUp+chunkId*chunkSize);
            if (controlBit){
                // store current state vector values in temp variables
                stateRealUp = qureg.stateVec[indexUp].real;
                stateImagUp = qureg.stateVec[indexUp].imag;

                stateRealLo = qureg.stateVec[indexLo].real;
                stateImagLo = qureg.stateVec[indexLo].imag;


                // state[indexUp] = u00 * state[indexUp] + u01 * state[indexLo]
                qureg.stateVec[indexUp].real = u.r0c0.real*stateRealUp - u.r0c0.imag*stateImagUp
                    + u.r0c1.real*stateRealLo - u.r0c1.imag*stateImagLo;
                qureg.stateVec[indexUp].imag = u.r0c0.real*stateImagUp + u.r0c0.imag*stateRealUp
                    + u.r0c1.real*stateImagLo + u.r0c1.imag*stateRealLo;

                // state[indexLo] = u10  * state[indexUp] + u11 * state[indexLo]
                qureg.stateVec[indexLo].real = u.r1c0.real*stateRealUp  - u.r1c0.imag*stateImagUp
                    + u.r1c1.real*stateRealLo  -  u.r1c1.imag*stateImagLo;
                qureg.stateVec[indexLo].imag = u.r1c0.real*stateImagUp + u.r1c0.imag*stateRealUp
                    + u.r1c1.real*stateImagLo + u.r1c1.imag*stateRealLo;
            }
        }
    }

}

/** Rotate a single qubit in the state vector of probability amplitudes, given two complex
 * numbers alpha and beta and a subset of the state vector with upper and lower block values
 * stored seperately. Only perform the rotation where the control qubit is one.
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] targetQubit qubit to rotate
 *  @param[in] controlQubit qubit to determine whether or not to perform a rotation
 *  @param[in] rot1 rotation angle
 *  @param[in] rot2 rotation angle
 *  @param[in] stateVecUp probability amplitudes in upper half of a block
 *  @param[in] stateVecLo probability amplitudes in lower half of a block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_controlledCompactUnitaryDistributed (Qureg qureg, const int controlQubit, const int targetQubit,
        Complex rot1, Complex rot2,
        Complex *stateVecUp,
        Complex *stateVecLo,
        Complex *stateVecOut)
{

    qreal   stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    int controlBit;

    qreal rot1Real=rot1.real, rot1Imag=rot1.imag;
    qreal rot2Real=rot2.real, rot2Imag=rot2.imag;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (stateVecUp,stateVecLo,stateVecOut, \
            rot1Real,rot1Imag, rot2Real,rot2Imag) \
    private  (thisTask,stateRealUp,stateImagUp,stateRealLo,stateImagLo,controlBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            controlBit = extractBit (controlQubit, thisTask+chunkId*chunkSize);
            if (controlBit){
                // store current state vector values in temp variables
                stateRealUp = stateVecUp[thisTask].real;
                stateImagUp = stateVecUp[thisTask].imag;

                stateRealLo = stateVecLo[thisTask].real;
                stateImagLo = stateVecLo[thisTask].imag;

                // state[indexUp] = alpha * state[indexUp] - conj(beta)  * state[indexLo]
                stateVecOut[thisTask].real = rot1Real*stateRealUp - rot1Imag*stateImagUp + rot2Real*stateRealLo + rot2Imag*stateImagLo;
                stateVecOut[thisTask].imag = rot1Real*stateImagUp + rot1Imag*stateRealUp + rot2Real*stateImagLo - rot2Imag*stateRealLo;
            }
        }
    }
}

/** Rotate a single qubit in the state vector of probability amplitudes, given two complex
 *  numbers alpha and beta and a subset of the state vector with upper and lower block values
 *  stored seperately. Only perform the rotation where the control qubit is one.
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] targetQubit qubit to rotate
 *  @param[in] controlQubit qubit to determine whether or not to perform a rotation
 *  @param[in] rot1 rotation angle
 *  @param[in] rot2 rotation angle
 *  @param[in] stateVecUp probability amplitudes in upper half of a block
 *  @param[in] stateVecLo probability amplitudes in lower half of a block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_controlledUnitaryDistributed (Qureg qureg, const int controlQubit, const int targetQubit,
        Complex rot1, Complex rot2,
        Complex *stateVecUp,
        Complex *stateVecLo,
        Complex *stateVecOut)
{

    qreal   stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    int controlBit;

    qreal rot1Real=rot1.real, rot1Imag=rot1.imag;
    qreal rot2Real=rot2.real, rot2Imag=rot2.imag;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (stateVecUp,stateVecLo,stateVecOut, \
            rot1Real,rot1Imag, rot2Real,rot2Imag) \
    private  (thisTask,stateRealUp,stateImagUp,stateRealLo,stateImagLo,controlBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            controlBit = extractBit (controlQubit, thisTask+chunkId*chunkSize);
            if (controlBit){
                // store current state vector values in temp variables
                stateRealUp = stateVecUp[thisTask].real;
                stateImagUp = stateVecUp[thisTask].imag;

                stateRealLo = stateVecLo[thisTask].real;
                stateImagLo = stateVecLo[thisTask].imag;

                stateVecOut[thisTask].real = rot1Real*stateRealUp - rot1Imag*stateImagUp
                    + rot2Real*stateRealLo - rot2Imag*stateImagLo;
                stateVecOut[thisTask].imag = rot1Real*stateImagUp + rot1Imag*stateRealUp
                    + rot2Real*stateImagLo + rot2Imag*stateRealLo;
            }
        }
    }
}

/** Apply a unitary operation to a single qubit in the state vector of probability amplitudes, given
 *  a subset of the state vector with upper and lower block values
 stored seperately. Only perform the rotation where all the control qubits are 1.
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] targetQubit qubit to rotate
 *  @param[in] controlQubit qubit to determine whether or not to perform a rotation
 *  @param[in] rot1 rotation angle
 *  @param[in] rot2 rotation angle
 *  @param[in] stateVecUp probability amplitudes in upper half of a block
 *  @param[in] stateVecLo probability amplitudes in lower half of a block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_multiControlledUnitaryDistributed (Qureg qureg,
        const int targetQubit,
        long long int mask,
        Complex rot1, Complex rot2,
        Complex *stateVecUp,
        Complex *stateVecLo,
        Complex *stateVecOut)
{

    qreal   stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    qreal rot1Real=rot1.real, rot1Imag=rot1.imag;
    qreal rot2Real=rot2.real, rot2Imag=rot2.imag;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (stateVecUp,stateVecLo,stateVecOut, \
            rot1Real,rot1Imag, rot2Real,rot2Imag, mask) \
    private  (thisTask,stateRealUp,stateImagUp,stateRealLo,stateImagLo)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            if (mask == (mask & (thisTask+chunkId*chunkSize)) ){
                // store current state vector values in temp variables
                stateRealUp = stateVecUp[thisTask].real;
                stateImagUp = stateVecUp[thisTask].imag;

                stateRealLo = stateVecLo[thisTask].real;
                stateImagLo = stateVecLo[thisTask].imag;

                stateVecOut[thisTask].real = rot1Real*stateRealUp - rot1Imag*stateImagUp
                    + rot2Real*stateRealLo - rot2Imag*stateImagLo;
                stateVecOut[thisTask].imag = rot1Real*stateImagUp + rot1Imag*stateRealUp
                    + rot2Real*stateImagLo + rot2Imag*stateRealLo;
            }
        }
    }
}

void statevec_pauliXLocal(Qureg qureg, const int targetQubit)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateImagUp;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            stateRealUp = qureg.stateVec[indexUp].real;
            stateImagUp = qureg.stateVec[indexUp].imag;

            qureg.stateVec[indexUp].real = qureg.stateVec[indexLo].real;
            qureg.stateVec[indexUp].imag = qureg.stateVec[indexLo].imag;

            qureg.stateVec[indexLo].real = stateRealUp;
            qureg.stateVec[indexLo].imag = stateImagUp;
        }
    }

}

/** Rotate a single qubit by {{0,1},{1,0}.
 *  Operate on a subset of the state vector with upper and lower block values
 *  stored seperately. This rotation is just swapping upper and lower values, and
 *  stateVecIn must already be the correct section for this chunk
 *
 *  @remarks Qubits are zero-based and the
 *  the first qubit is the rightmost
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] targetQubit qubit to rotate
 *  @param[in] stateVecIn probability amplitudes in lower or upper half of a block depending on chunkId
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_pauliXDistributed (Qureg qureg, const int targetQubit,
        Complex *stateVecIn,
        Complex *stateVecOut)
{

    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;


# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (quregInIn,stateVecOut) \
    private  (thisTask)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            stateVecOut[thisTask].real = stateVecIn[thisTask].real;
            stateVecOut[thisTask].imag = stateVecIn[thisTask].imag;
        }
    }
}

void statevec_controlledNotLocal(Qureg qureg, const int controlQubit, const int targetQubit)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateImagUp;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    int controlBit;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp,controlBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            controlBit = extractBit(controlQubit, indexUp+chunkId*chunkSize);
            if (controlBit){
                stateRealUp = qureg.stateVec[indexUp].real;
                stateImagUp = qureg.stateVec[indexUp].imag;

                qureg.stateVec[indexUp].real = qureg.stateVec[indexLo].real;
                qureg.stateVec[indexUp].imag = qureg.stateVec[indexLo].imag;

                qureg.stateVec[indexLo].real = stateRealUp;
                qureg.stateVec[indexLo].imag = stateImagUp;
            }
        }
    }
}

/** Rotate a single qubit by {{0,1},{1,0}.
 *  Operate on a subset of the state vector with upper and lower block values
 *  stored seperately. This rotation is just swapping upper and lower values, and
 *  stateVecIn must already be the correct section for this chunk. Only perform the rotation
 *  for elements where controlQubit is one.
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] targetQubit qubit to rotate
 *  @param[in] stateVecIn probability amplitudes in lower or upper half of a block depending on chunkId
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_controlledNotDistributed (Qureg qureg, const int controlQubit, const int targetQubit,
        Complex *stateVecIn,
        Complex *stateVecOut)
{

    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    int controlBit;


# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (quregInIn,stateVecOut) \
    private  (thisTask,controlBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            controlBit = extractBit (controlQubit, thisTask+chunkId*chunkSize);
            if (controlBit){
                stateVecOut[thisTask].real = stateVecIn[thisTask].real;
                stateVecOut[thisTask].imag = stateVecIn[thisTask].imag;
            }
        }
    }
}

void statevec_pauliYLocal(Qureg qureg, const int targetQubit, const int conjFac)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateImagUp;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            stateRealUp = qureg.stateVec[indexUp].real;
            stateImagUp = qureg.stateVec[indexUp].imag;

            qureg.stateVec[indexUp].imag = conjFac *  qureg.stateVec[indexLo].real;
            qureg.stateVec[indexUp].real = conjFac * -qureg.stateVec[indexLo].imag;
            qureg.stateVec[indexLo].real = conjFac * -stateImagUp;
            qureg.stateVec[indexLo].imag = conjFac * stateRealUp;
        }
    }
}

/** Rotate a single qubit by +-{{0,-i},{i,0}.
 *  Operate on a subset of the state vector with upper and lower block values
 *  stored seperately. This rotation is just swapping upper and lower values, and
 *  stateVecIn must already be the correct section for this chunk
 *
 *  @remarks Qubits are zero-based and the
 *  the first qubit is the rightmost
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] targetQubit qubit to rotate
 *  @param[in] stateVecIn probability amplitudes in lower or upper half of a block depending on chunkId
 *  @param[in] updateUpper flag, 1: updating upper values, 0: updating lower values in block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_pauliYDistributed(Qureg qureg, const int targetQubit,
        Complex *stateVecIn,
        Complex *stateVecOut,
        int updateUpper, const int conjFac)
{

    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;


    int realSign=1, imagSign=1;
    if (updateUpper) imagSign=-1;
    else realSign = -1;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (quregInIn,stateVecOut,realSign,imagSign) \
    private  (thisTask)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            stateVecOut[thisTask].real = conjFac * realSign * stateVecIn[thisTask].imag;
            stateVecOut[thisTask].imag = conjFac * imagSign * stateVecIn[thisTask].real;
        }
    }
}




void statevec_controlledPauliYLocal(Qureg qureg, const int controlQubit, const int targetQubit, const int conjFac)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateImagUp;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    int controlBit;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp,controlBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            controlBit = extractBit(controlQubit, indexUp+chunkId*chunkSize);
            if (controlBit){
                stateRealUp = qureg.stateVec[indexUp].real;
                stateImagUp = qureg.stateVec[indexUp].imag;

                // update under +-{{0, -i}, {i, 0}}
                qureg.stateVec[indexUp].imag = conjFac * qureg.stateVec[indexLo].real;
                qureg.stateVec[indexUp].real = conjFac * -qureg.stateVec[indexLo].imag;
                qureg.stateVec[indexLo].real = conjFac * -stateImagUp;
                qureg.stateVec[indexLo].imag = conjFac * stateRealUp;
            }
        }
    }
}


void statevec_controlledPauliYDistributed (Qureg qureg, const int controlQubit, const int targetQubit,
        Complex *stateVecIn,
        Complex *stateVecOut, const int conjFac)
{

    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;
    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    int controlBit;


# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (quregInIn,stateVecOut) \
    private  (thisTask,controlBit)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            controlBit = extractBit (controlQubit, thisTask+chunkId*chunkSize);
            if (controlBit){
                stateVecOut[thisTask].real = conjFac *  stateVecIn[thisTask].imag;
                stateVecOut[thisTask].imag = conjFac * -stateVecIn[thisTask].real;
            }
        }
    }
}







void statevec_hadamardLocal(Qureg qureg, const int targetQubit)
{
    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, // current block
         indexUp,indexLo;    // current index and corresponding index in lower half block

    qreal stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    // set dimensions
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    // Can't use qureg.stateVec as a private OMP var

    qreal recRoot2 = 1.0/sqrt(2);

# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (qureg, sizeBlock,sizeHalfBlock, recRoot2) \
    private  (thisTask,thisBlock ,indexUp,indexLo, stateRealUp,stateImagUp,stateRealLo,stateImagLo)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            thisBlock   = thisTask / sizeHalfBlock;
            indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
            indexLo     = indexUp + sizeHalfBlock;

            stateRealUp = qureg.stateVec[indexUp].real;
            stateImagUp = qureg.stateVec[indexUp].imag;

            stateRealLo = qureg.stateVec[indexLo].real;
            stateImagLo = qureg.stateVec[indexLo].imag;

            qureg.stateVec[indexUp].real = recRoot2*(stateRealUp + stateRealLo);
            qureg.stateVec[indexUp].imag = recRoot2*(stateImagUp + stateImagLo);

            qureg.stateVec[indexLo].real = recRoot2*(stateRealUp - stateRealLo);
            qureg.stateVec[indexLo].imag = recRoot2*(stateImagUp - stateImagLo);
        }
    }
}

/** Rotate a single qubit by {{1,1},{1,-1}}/sqrt2.
 *  Operate on a subset of the state vector with upper and lower block values
 *  stored seperately. This rotation is just swapping upper and lower values, and
 *  stateVecIn must already be the correct section for this chunk
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] targetQubit qubit to rotate
 *  @param[in] stateVecIn probability amplitudes in lower or upper half of a block depending on chunkId
 *  @param[in] updateUpper flag, 1: updating upper values, 0: updating lower values in block
 *  @param[out] stateVecOut array section to update (will correspond to either the lower or upper half of a block)
 */
void statevec_hadamardDistributed(Qureg qureg, const int targetQubit,
        Complex *stateVecUp,
        Complex *stateVecLo,
        Complex *stateVecOut,
        int updateUpper)
{

    qreal   stateRealUp,stateRealLo,stateImagUp,stateImagLo;
    long long int thisTask;
    const long long int numTasks=qureg.numAmpsPerChunk;

    int sign;
    if (updateUpper) sign=1;
    else sign=-1;

    qreal recRoot2 = 1.0/sqrt(2);


# ifdef _OPENMP
# pragma omp parallel \
    default  (none) \
    shared   (stateVecUp,stateVecLo,stateVecOut, \
            recRoot2, sign) \
    private  (thisTask,stateRealUp,stateImagUp,stateRealLo,stateImagLo)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            // store current state vector values in temp variables
            stateRealUp = stateVecUp[thisTask].real;
            stateImagUp = stateVecUp[thisTask].imag;

            stateRealLo = stateVecLo[thisTask].real;
            stateImagLo = stateVecLo[thisTask].imag;

            stateVecOut[thisTask].real = recRoot2*(stateRealUp + sign*stateRealLo);
            stateVecOut[thisTask].imag = recRoot2*(stateImagUp + sign*stateImagLo);
        }
    }
}

void statevec_phaseShiftByTerm (Qureg qureg, const int targetQubit, Complex term)
{
    long long int index;
    long long int stateVecSize;
    int targetBit;

    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    // dimension of the state vector
    stateVecSize = qureg.numAmpsPerChunk;

    qreal stateRealLo, stateImagLo;
    const qreal cosAngle = term.real;
    const qreal sinAngle = term.imag;

# ifdef _OPENMP
# pragma omp parallel for \
    default  (none)              \
    shared   (qureg, stateVecSize) \
    private  (index,targetBit,stateRealLo,stateImagLo)             \
    schedule (static)
# endif
    for (index=0; index<stateVecSize; index++) {

        // update the coeff of the |1> state of the target qubit
        targetBit = extractBit (targetQubit, index+chunkId*chunkSize);
        if (targetBit) {

            stateRealLo = qureg.stateVec[index].real;
            stateImagLo = qureg.stateVec[index].imag;

            qureg.stateVec[index].real = cosAngle*stateRealLo - sinAngle*stateImagLo;
            qureg.stateVec[index].imag = sinAngle*stateRealLo + cosAngle*stateImagLo;
        }
    }
}

void statevec_controlledPhaseShift (Qureg qureg, const int idQubit1, const int idQubit2, qreal angle)
{
    long long int index;
    long long int stateVecSize;
    int bit1, bit2;

    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    // dimension of the state vector
    stateVecSize = qureg.numAmpsPerChunk;

    qreal stateRealLo, stateImagLo;
    const qreal cosAngle = cos(angle);
    const qreal sinAngle = sin(angle);

# ifdef _OPENMP
# pragma omp parallel for \
    default  (none)              \
    shared   (qureg, stateVecSize) \
    private  (index,bit1,bit2,stateRealLo,stateImagLo)             \
    schedule (static)
# endif
    for (index=0; index<stateVecSize; index++) {
        bit1 = extractBit (idQubit1, index+chunkId*chunkSize);
        bit2 = extractBit (idQubit2, index+chunkId*chunkSize);
        if (bit1 && bit2) {

            stateRealLo = qureg.stateVec[index].real;
            stateImagLo = qureg.stateVec[index].imag;

            qureg.stateVec[index].real = cosAngle*stateRealLo - sinAngle*stateImagLo;
            qureg.stateVec[index].imag = sinAngle*stateRealLo + cosAngle*stateImagLo;
        }
    }
}

void statevec_multiControlledPhaseShift(Qureg qureg, int *controlQubits, int numControlQubits, qreal angle)
{
    long long int index;
    long long int stateVecSize;

    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    long long int mask=0;
    for (int i=0; i<numControlQubits; i++)
        mask = mask | (1LL<<controlQubits[i]);

    stateVecSize = qureg.numAmpsPerChunk;

    qreal stateRealLo, stateImagLo;
    const qreal cosAngle = cos(angle);
    const qreal sinAngle = sin(angle);

# ifdef _OPENMP
# pragma omp parallel \
    default  (none)              \
    shared   (qureg, stateVecSize, mask) \
    private  (index, stateRealLo, stateImagLo)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<stateVecSize; index++) {
            if (mask == (mask & (index+chunkId*chunkSize)) ){

                stateRealLo = qureg.stateVec[index].real;
                stateImagLo = qureg.stateVec[index].imag;

                qureg.stateVec[index].real = cosAngle*stateRealLo - sinAngle*stateImagLo;
                qureg.stateVec[index].imag = sinAngle*stateRealLo + cosAngle*stateImagLo;
            }
        }
    }
}


qreal densmatr_findProbabilityOfZeroLocal(Qureg qureg, const int measureQubit) {

    // computes first local index containing a diagonal element
    long long int localNumAmps = qureg.numAmpsPerChunk;
    long long int densityDim = (1LL << qureg.numQubitsRepresented);
    long long int diagSpacing = 1LL + densityDim;
    long long int maxNumDiagsPerChunk = 1 + localNumAmps / diagSpacing;
    long long int numPrevDiags = (qureg.chunkId>0)? 1+(qureg.chunkId*localNumAmps)/diagSpacing : 0;
    long long int globalIndNextDiag = diagSpacing * numPrevDiags;
    long long int localIndNextDiag = globalIndNextDiag % localNumAmps;

    // computes how many diagonals are contained in this chunk
    long long int numDiagsInThisChunk = maxNumDiagsPerChunk;
    if (localIndNextDiag + (numDiagsInThisChunk-1)*diagSpacing >= localNumAmps)
        numDiagsInThisChunk -= 1;

    long long int visitedDiags;     // number of visited diagonals in this chunk so far
    long long int basisStateInd;    // current diagonal index being considered
    long long int index;            // index in the local chunk

    qreal zeroProb = 0;

# ifdef _OPENMP
# pragma omp parallel \
    shared    (qureg, localIndNextDiag, numPrevDiags, diagSpacing, numDiagsInThisChunk) \
    private   (visitedDiags, basisStateInd, index) \
    reduction ( +:zeroProb )
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
        // sums the diagonal elems of the density matrix where measureQubit=0
        for (visitedDiags = 0; visitedDiags < numDiagsInThisChunk; visitedDiags++) {

            basisStateInd = numPrevDiags + visitedDiags;
            index = localIndNextDiag + diagSpacing * visitedDiags;

            if (extractBit(measureQubit, basisStateInd) == 0)
                zeroProb += qureg.stateVec[index].real; // assume imag[diagonls] ~ 0

        }
    }

    return zeroProb;
}

/** Measure the total probability of a specified qubit being in the zero state across all amplitudes in this chunk.
 *  Size of regions to skip is less than the size of one chunk.
 *
 *  @param[in] qureg object representing the set of qubits
 *  @param[in] measureQubit qubit to measure
 *  @return probability of qubit measureQubit being zero
 */
qreal statevec_findProbabilityOfZeroLocal (Qureg qureg,
        const int measureQubit)
{
    // ----- sizes
    long long int sizeBlock,                                  // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                  // current block
         index;                                               // current index for first half block
    // ----- measured probability
    qreal   totalProbability;                                  // probability (returned) value
    // ----- temp variables
    long long int thisTask;
    long long int numTasks=qureg.numAmpsPerChunk>>1;

    // ---------------------------------------------------------------- //
    //            dimensions                                            //
    // ---------------------------------------------------------------- //
    sizeHalfBlock = 1LL << (measureQubit);                       // number of state vector elements to sum,
    // and then the number to skip
    sizeBlock     = 2LL * sizeHalfBlock;                         // size of blocks (pairs of measure and skip entries)

    // initialise returned value
    totalProbability = 0.0;


# ifdef _OPENMP
# pragma omp parallel \
    shared    (qureg, numTasks,sizeBlock,sizeHalfBlock) \
    private   (thisTask,thisBlock,index) \
    reduction ( +:totalProbability )
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            thisBlock = thisTask / sizeHalfBlock;
            index     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;

            totalProbability += qureg.stateVec[index].real*qureg.stateVec[index].real
                + qureg.stateVec[index].imag*qureg.stateVec[index].imag;
        }
    }
    return totalProbability;
}

/** Measure the probability of a specified qubit being in the zero state across all amplitudes held in this chunk.
 * Size of regions to skip is a multiple of chunkSize.
 * The results are communicated and aggregated by the caller
 *
 *  @param[in] qureg object representing the set of qubits
 *  @param[in] measureQubit qubit to measure
 *  @return probability of qubit measureQubit being zero
 */
qreal statevec_findProbabilityOfZeroDistributed (Qureg qureg,
        const int measureQubit)
{
    // ----- measured probability
    qreal   totalProbability;                                  // probability (returned) value
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    long long int numTasks=qureg.numAmpsPerChunk;

    // ---------------------------------------------------------------- //
    //            find probability                                      //
    // ---------------------------------------------------------------- //

    // initialise returned value
    totalProbability = 0.0;


# ifdef _OPENMP
# pragma omp parallel \
    shared    (qureg, numTasks) \
    private   (thisTask) \
    reduction ( +:totalProbability )
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            totalProbability += qureg.stateVec[thisTask].real*qureg.stateVec[thisTask].real
                + qureg.stateVec[thisTask].imag*qureg.stateVec[thisTask].imag;
        }
    }

    return totalProbability;
}



void statevec_controlledPhaseFlip (Qureg qureg, const int idQubit1, const int idQubit2)
{
    long long int index;
    long long int stateVecSize;
    int bit1, bit2;

    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    // dimension of the state vector
    stateVecSize = qureg.numAmpsPerChunk;

# ifdef _OPENMP
# pragma omp parallel for \
    default  (none)              \
    shared   (qureg, stateVecSize) \
    private  (index,bit1,bit2)             \
    schedule (static)
# endif
    for (index=0; index<stateVecSize; index++) {
        bit1 = extractBit (idQubit1, index+chunkId*chunkSize);
        bit2 = extractBit (idQubit2, index+chunkId*chunkSize);
        if (bit1 && bit2) {
            qureg.stateVec[index].real = - qureg.stateVec[index].real;
            qureg.stateVec[index].imag = - qureg.stateVec[index].imag;
        }
    }
}

void statevec_multiControlledPhaseFlip(Qureg qureg, int *controlQubits, int numControlQubits)
{
    long long int index;
    long long int stateVecSize;

    const long long int chunkSize=qureg.numAmpsPerChunk;
    const long long int chunkId=qureg.chunkId;

    long long int mask=0;
    for (int i=0; i<numControlQubits; i++)
        mask = mask | (1LL<<controlQubits[i]);

    stateVecSize = qureg.numAmpsPerChunk;

# ifdef _OPENMP
# pragma omp parallel \
    default  (none)              \
    shared   (qureg, stateVecSize, mask ) \
    private  (index)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule (static)
# endif
        for (index=0; index<stateVecSize; index++) {
            if (mask == (mask & (index+chunkId*chunkSize)) ){
                qureg.stateVec[index].real = - qureg.stateVec[index].real;
                qureg.stateVec[index].imag = - qureg.stateVec[index].imag;
            }
        }
    }
}

/** Update the state vector to be consistent with measuring measureQubit=0 if outcome=0 and measureQubit=1
 *  if outcome=1.
 *  Performs an irreversible change to the state vector: it updates the vector according
 *  to the event that an outcome have been measured on the qubit indicated by measureQubit (where
 *  this label starts from 0, of course). It achieves this by setting all inconsistent
 *  amplitudes to 0 and
 *  then renormalising based on the total probability of measuring measureQubit=0 or 1 according to the
 *  value of outcome.
 *  In the local version, one or more blocks (with measureQubit=0 in the first half of the block and
 *  measureQubit=1 in the second half of the block) fit entirely into one chunk.
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] measureQubit qubit to measure
 *  @param[in] totalProbability probability of qubit measureQubit being either zero or one
 *  @param[in] outcome to measure the probability of and set the state to -- either zero or one
 */
void statevec_collapseToKnownProbOutcomeLocal(Qureg qureg, int measureQubit, int outcome, qreal totalProbability)
{
    // ----- sizes
    long long int sizeBlock,                                  // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                  // current block
         index;                                               // current index for first half block
    // ----- measured probability
    qreal   renorm;                                            // probability (returned) value
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    // (good for shared memory parallelism)
    long long int numTasks=qureg.numAmpsPerChunk>>1;

    // ---------------------------------------------------------------- //
    //            dimensions                                            //
    // ---------------------------------------------------------------- //
    sizeHalfBlock = 1LL << (measureQubit);                       // number of state vector elements to sum,
    // and then the number to skip
    sizeBlock     = 2LL * sizeHalfBlock;                         // size of blocks (pairs of measure and skip entries)

    renorm=1/sqrt(totalProbability);


# ifdef _OPENMP
# pragma omp parallel \
    default (none) \
    shared    (qureg, numTasks,sizeBlock,sizeHalfBlock,renorm,outcome) \
    private   (thisTask,thisBlock,index)
# endif
    {
        if (outcome==0){
            // measure qubit is 0
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
            for (thisTask=0; thisTask<numTasks; thisTask++) {
                thisBlock = thisTask / sizeHalfBlock;
                index     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
                qureg.stateVec[index].real = qureg.stateVec[index].real*renorm;
                qureg.stateVec[index].imag = qureg.stateVec[index].imag*renorm;

                qureg.stateVec[index+sizeHalfBlock].real=0;
                qureg.stateVec[index+sizeHalfBlock].imag=0;
            }
        } else {
            // measure qubit is 1
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
            for (thisTask=0; thisTask<numTasks; thisTask++) {
                thisBlock = thisTask / sizeHalfBlock;
                index     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
                qureg.stateVec[index].real=0;
                qureg.stateVec[index].imag=0;

                qureg.stateVec[index+sizeHalfBlock].real = qureg.stateVec[index+sizeHalfBlock].real*renorm;
                qureg.stateVec[index+sizeHalfBlock].imag = qureg.stateVec[index+sizeHalfBlock].imag*renorm;
            }
        }
    }

}

/** Renormalise parts of the state vector where measureQubit=0 or 1, based on the total probability of that qubit being
 *  in state 0 or 1.
 *  Measure in Zero performs an irreversible change to the state vector: it updates the vector according
 *  to the event that the value 'outcome' has been measured on the qubit indicated by measureQubit (where
 *  this label starts from 0, of course). It achieves this by setting all inconsistent amplitudes to 0 and
 *  then renormalising based on the total probability of measuring measureQubit=0 if outcome=0 and
 *  measureQubit=1 if outcome=1.
 *  In the distributed version, one block (with measureQubit=0 in the first half of the block and
 *  measureQubit=1 in the second half of the block) is spread over multiple chunks, meaning that each chunks performs
 *  only renormalisation or only setting amplitudes to 0. This function handles the renormalisation.
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] measureQubit qubit to measure
 *  @param[in] totalProbability probability of qubit measureQubit being zero
 */
void statevec_collapseToKnownProbOutcomeDistributedRenorm (Qureg qureg, const int measureQubit, const qreal totalProbability)
{
    // ----- temp variables
    long long int thisTask;
    long long int numTasks=qureg.numAmpsPerChunk;

    qreal renorm=1/sqrt(totalProbability);


# ifdef _OPENMP
# pragma omp parallel \
    shared    (qureg, numTasks) \
    private   (thisTask)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            qureg.stateVec[thisTask].real = qureg.stateVec[thisTask].real*renorm;
            qureg.stateVec[thisTask].imag = qureg.stateVec[thisTask].imag*renorm;
        }
    }
}

/** Set all amplitudes in one chunk to 0.
 *  Measure in Zero performs an irreversible change to the state vector: it updates the vector according
 *  to the event that a zero have been measured on the qubit indicated by measureQubit (where
 *  this label starts from 0, of course). It achieves this by setting all inconsistent amplitudes to 0 and
 *  then renormalising based on the total probability of measuring measureQubit=0 or 1.
 *  In the distributed version, one block (with measureQubit=0 in the first half of the block and
 *  measureQubit=1 in the second half of the block) is spread over multiple chunks, meaning that each chunks performs
 *  only renormalisation or only setting amplitudes to 0. This function handles setting amplitudes to 0.
 *
 *  @param[in,out] qureg object representing the set of qubits
 *  @param[in] measureQubit qubit to measure
 */
void statevec_collapseToOutcomeDistributedSetZero(Qureg qureg)
{
    // ----- temp variables
    long long int thisTask;
    long long int numTasks=qureg.numAmpsPerChunk;

    // ---------------------------------------------------------------- //
    //            find probability                                      //
    // ---------------------------------------------------------------- //


# ifdef _OPENMP
# pragma omp parallel \
    shared    (qureg, numTasks) \
    private   (thisTask)
# endif
    {
# ifdef _OPENMP
# pragma omp for schedule  (static)
# endif
        for (thisTask=0; thisTask<numTasks; thisTask++) {
            qureg.stateVec[thisTask].real = 0;
            qureg.stateVec[thisTask].imag = 0;
        }
    }
}
