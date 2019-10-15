// Distributed under MIT licence. See https://github.com/QuEST-Kit/QuEST/blob/master/LICENCE.txt for details

/** @file
 * An implementation of the backend in ../QuEST_internal.h for a GPU environment.
 */

# include "QuEST.h"
# include "QuEST_precision.h"
# include "QuEST_internal.h"    // purely to resolve getQuESTDefaultSeedKey
# include "mt19937ar.h"

# include <stdlib.h>
# include <stdio.h>
# include <math.h>

# define REDUCE_SHARED_SIZE 512
# define DEBUG 0


static __device__ int extractBit (int locationOfBitFromRight, long long int theEncodedNumber)
{
    return (theEncodedNumber & ( 1LL << locationOfBitFromRight )) >> locationOfBitFromRight;
}

#ifdef __cplusplus
extern "C" {
#endif






void statevec_setAmps(Qureg qureg, long long int startInd, qreal* reals, qreal* imags, long long int numAmps) {

    cudaDeviceSynchronize();

    cudaMemcpy2D (qureg.deviceStateVec + startInd,
                  2 * sizeof(reals[0]),
                  reals,
                  sizeof(reals[0]),
                  sizeof(reals[0]), numAmps,
                  cudaMemcpyHostToDevice);
    cudaMemcpy2D (qureg.deviceStateVec + startInd + 1,
                  2 * sizeof(imags[0]),
                  imags,
                  1 * sizeof(imags[0]),
                  sizeof(imags[0]), numAmps,
                  cudaMemcpyHostToDevice);
    /* cudaMemcpy( */
    /*     qureg.deviceStateVec.real + startInd,  */
    /*     reals, */
    /*     numAmps * sizeof(*(qureg.deviceStateVec.real)),  */
    /*     cudaMemcpyHostToDevice); */
    /* cudaMemcpy( */
    /*     qureg.deviceStateVec.imag + startInd, */
    /*     imags, */
    /*     numAmps * sizeof(*(qureg.deviceStateVec.real)),  */
    /*     cudaMemcpyHostToDevice); */
}


/** works for both statevectors and density matrices */
void statevec_cloneQureg(Qureg targetQureg, Qureg copyQureg) {

    // copy copyQureg's GPU statevec to targetQureg's GPU statevec
    cudaDeviceSynchronize();
    cudaMemcpy(
        targetQureg.deviceStateVec,
        copyQureg.deviceStateVec,
        targetQureg.numAmpsPerChunk*sizeof(*(targetQureg.deviceStateVec)),
        cudaMemcpyDeviceToDevice);
}

__global__ void densmatr_initPureStateKernel(
    long long int numPureAmps, Complex *targetVec, Complex *copyVec)
    /* qreal *targetVecReal, qreal *targetVecImag,  */
    /* qreal *copyVecReal, qreal *copyVecImag)  */
{
    // this is a particular index of the pure copyQureg
    long long int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=numPureAmps) return;

    qreal realRow = copyVec[index].real;
    qreal imagRow = copyVec[index].imag;
    for (long long int col=0; col < numPureAmps; col++) {
        qreal realCol =   copyVec[col].real;
        qreal imagCol = - copyVec[col].imag; // minus for conjugation
        targetVec[col*numPureAmps + index].real = realRow*realCol - imagRow*imagCol;
        targetVec[col*numPureAmps + index].imag = realRow*imagCol + imagRow*realCol;
    }
}

void densmatr_initPureState(Qureg targetQureg, Qureg copyQureg)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(copyQureg.numAmpsPerChunk)/threadsPerCUDABlock);
    densmatr_initPureStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        copyQureg.numAmpsPerChunk,
        targetQureg.deviceStateVec, copyQureg.deviceStateVec);
}

__global__ void densmatr_initPlusStateKernel(long long int stateVecSize, qreal probFactor, Complex *stateVec){
    long long int index;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    stateVec[index].real = probFactor;
    stateVec[index].imag = 0.0;
}

void densmatr_initPlusState(Qureg qureg)
{
    qreal probFactor = 1.0/((qreal) (1LL << qureg.numQubitsRepresented));
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    densmatr_initPlusStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        qureg.numAmpsPerChunk,
        probFactor,
        qureg.deviceStateVec);
}

__global__ void densmatr_initClassicalStateKernel(
    long long int densityNumElems,
    Complex *densityVec,
    long long int densityInd)
{
    // initialise the state to all zeros
    long long int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= densityNumElems) return;

    densityVec[index].real = 0.0;
    densityVec[index].imag = 0.0;

    if (index==densityInd){
        // classical state has probability 1
        densityVec[densityInd].real = 1.0;
        densityVec[densityInd].imag = 0.0;
    }
}

void densmatr_initClassicalState(Qureg qureg, long long int stateInd)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);

    // index of the desired state in the flat density matrix
    long long int densityDim = 1LL << qureg.numQubitsRepresented;
    long long int densityInd = (densityDim + 1)*stateInd;

    // identical to pure version
    densmatr_initClassicalStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        qureg.numAmpsPerChunk,
        qureg.deviceStateVec, densityInd);
}

void statevec_createQureg(Qureg *qureg, int numQubits, QuESTEnv env)
{
    // allocate CPU memory
    long long int numAmps = 1L << numQubits;
    long long int numAmpsPerRank = numAmps/env.numRanks;
    qureg->stateVec = (Complex*) malloc(numAmpsPerRank * sizeof(qureg->stateVec));
    if (env.numRanks>1){
        qureg->pairStateVec = (Complex*) malloc(numAmpsPerRank * sizeof(qureg->pairStateVec));
    }

    // check cpu memory allocation was successful
    if ( (!(qureg->stateVec)) && numAmpsPerRank ) {
        printf("Could not allocate memory!\n");
        exit (EXIT_FAILURE);
    }
    if ( env.numRanks>1 && (!(qureg->pairStateVec)) && numAmpsPerRank ) {
        printf("Could not allocate memory!\n");
        exit (EXIT_FAILURE);
    }

    qureg->numQubitsInStateVec = numQubits;
    qureg->numAmpsPerChunk = numAmpsPerRank;
    qureg->numAmpsTotal = numAmps;
    qureg->chunkId = env.rank;
    qureg->numChunks = env.numRanks;
    qureg->isDensityMatrix = 0;

    // allocate GPU memory
    cudaMalloc(&(qureg->deviceStateVec), qureg->numAmpsPerChunk*sizeof(*(qureg->deviceStateVec)));
    cudaMalloc(&(qureg->firstLevelReduction), ceil(qureg->numAmpsPerChunk/(qreal)REDUCE_SHARED_SIZE)*sizeof(qreal));
    cudaMalloc(&(qureg->secondLevelReduction), ceil(qureg->numAmpsPerChunk/(qreal)(REDUCE_SHARED_SIZE*REDUCE_SHARED_SIZE))*
            sizeof(qreal));

    // check gpu memory allocation was successful
    if (!(qureg->deviceStateVec)) {
        printf("Could not allocate memory on GPU!\n");
        exit (EXIT_FAILURE);
    }

}

void statevec_destroyQureg(Qureg qureg, QuESTEnv env)
{
    // Free CPU memory
    free(qureg.stateVec);
    if (env.numRanks>1){
        free(qureg.pairStateVec);
    }

    // Free GPU memory
    cudaFree(qureg.deviceStateVec);
}

int GPUExists(void){
    int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess) deviceCount = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) { /* 9999 means emulation only */
            ++gpuDeviceCount;
        }
    }
    if (gpuDeviceCount) return 1;
    else return 0;
}

QuESTEnv createQuESTEnv(void) {
    // init MPI environment
    if (!GPUExists()){
        printf("Trying to run GPU code with no GPU available\n");
        exit(EXIT_FAILURE);
    }

    QuESTEnv env;
    env.rank=0;
    env.numRanks=1;

    seedQuESTDefault();

    return env;
}

void syncQuESTEnv(QuESTEnv env){
    cudaDeviceSynchronize();
}

int syncQuESTSuccess(int successCode){
    return successCode;
}

void destroyQuESTEnv(QuESTEnv env){
    // MPI finalize goes here in MPI version. Call this function anyway for consistency
}

void reportQuESTEnv(QuESTEnv env){
    printf("EXECUTION ENVIRONMENT:\n");
    printf("Running locally on one node with GPU\n");
    printf("Number of ranks is %d\n", env.numRanks);
# ifdef _OPENMP
    printf("OpenMP enabled\n");
    printf("Number of threads available is %d\n", omp_get_max_threads());
# else
    printf("OpenMP disabled\n");
# endif
}

void getEnvironmentString(QuESTEnv env, Qureg qureg, char str[200]){
    sprintf(str, "%dqubits_GPU_noMpi_noOMP", qureg.numQubitsInStateVec);
}

void copyStateToGPU(Qureg qureg)
{
    if (DEBUG) printf("Copying data to GPU\n");
    cudaMemcpy(qureg.deviceStateVec, qureg.stateVec,
            qureg.numAmpsPerChunk*sizeof(*(qureg.deviceStateVec)), cudaMemcpyHostToDevice);

    /* cudaMemcpy(qureg.deviceStateVec.real, qureg.stateVec.real, */
    /*         qureg.numAmpsPerChunk*sizeof(*(qureg.deviceStateVec.real)), cudaMemcpyHostToDevice); */
    /* cudaMemcpy(qureg.deviceStateVec.real, qureg.stateVec.real, */
    /*         qureg.numAmpsPerChunk*sizeof(*(qureg.deviceStateVec.real)), cudaMemcpyHostToDevice); */
    /* cudaMemcpy(qureg.deviceStateVec.imag, qureg.stateVec.imag, */
    /*         qureg.numAmpsPerChunk*sizeof(*(qureg.deviceStateVec.imag)), cudaMemcpyHostToDevice); */
    /* cudaMemcpy(qureg.deviceStateVec.imag, qureg.stateVec.imag, */
    /*         qureg.numAmpsPerChunk*sizeof(*(qureg.deviceStateVec.imag)), cudaMemcpyHostToDevice); */
    if (DEBUG) printf("Finished copying data to GPU\n");
}

void copyStateFromGPU(Qureg qureg)
{
    cudaDeviceSynchronize();
    if (DEBUG) printf("Copying data from GPU\n");
    cudaMemcpy(qureg.stateVec, qureg.deviceStateVec,
            qureg.numAmpsPerChunk*sizeof(*(qureg.deviceStateVec)), cudaMemcpyDeviceToHost);

    /* cudaMemcpy(qureg.stateVec.real, qureg.deviceStateVec.real, */
    /*         qureg.numAmpsPerChunk*sizeof(*(qureg.deviceStateVec.real)), cudaMemcpyDeviceToHost); */
    /* cudaMemcpy(qureg.stateVec.imag, qureg.deviceStateVec.imag, */
    /*         qureg.numAmpsPerChunk*sizeof(*(qureg.deviceStateVec.imag)), cudaMemcpyDeviceToHost); */
    if (DEBUG) printf("Finished copying data from GPU\n");
}

/** Print the current state vector of probability amplitudes for a set of qubits to standard out.
  For debugging purposes. Each rank should print output serially. Only print output for systems <= 5 qubits
 */
void statevec_reportStateToScreen(Qureg qureg, QuESTEnv env, int reportRank){
    long long int index;
    int rank;
    copyStateFromGPU(qureg);
    if (qureg.numQubitsInStateVec<=5){
        for (rank=0; rank<qureg.numChunks; rank++){
            if (qureg.chunkId==rank){
                if (reportRank) {
                    printf("Reporting state from rank %d [\n", qureg.chunkId);
                    //printf("\trank, index, real, imag\n");
                    printf("real, imag\n");
                } else if (rank==0) {
                    printf("Reporting state [\n");
                    printf("real, imag\n");
                }

                for(index=0; index<qureg.numAmpsPerChunk; index++){
                    printf(REAL_STRING_FORMAT ", " REAL_STRING_FORMAT "\n", qureg.stateVec[index].real, qureg.stateVec[index].imag);
                }
                if (reportRank || rank==qureg.numChunks-1) printf("]\n");
            }
            syncQuESTEnv(env);
        }
    }
}

qreal statevec_getRealAmp(Qureg qureg, long long int index){
    qreal el=0;
    cudaMemcpy(&el, &(qureg.deviceStateVec[index].real),
            sizeof(*(qureg.deviceStateVec[index].real)), cudaMemcpyDeviceToHost);
    return el;
}

qreal statevec_getImagAmp(Qureg qureg, long long int index){
    qreal el=0;
    cudaMemcpy(&el, &(qureg.deviceStateVec[index].imag),
            sizeof(*(qureg.deviceStateVec[index].imag)), cudaMemcpyDeviceToHost);
    return el;
}

__global__ void statevec_initZeroStateKernel(long long int stateVecSize, Complex *stateVec){
    long long int index;

    // initialise the state to |0000..0000>
    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;
    stateVec[index].real = 0.0;
    stateVec[index].imag = 0.0;

    if (index==0){
        // zero state |0000..0000> has probability 1
        stateVec[0].real = 1.0;
        stateVec[0].imag = 0.0;
    }
}

void statevec_initZeroState(Qureg qureg)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_initZeroStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        qureg.numAmpsPerChunk,
        qureg.deviceStateVec);
}

__global__ void statevec_initPlusStateKernel(long long int stateVecSize, Complex *stateVec){
    long long int index;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    qreal normFactor = 1.0/sqrt((qreal)stateVecSize);
    stateVec[index].real = normFactor;
    stateVec[index].imag = 0.0;
}

void statevec_initPlusState(Qureg qureg)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_initPlusStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        qureg.numAmpsPerChunk,
        qureg.deviceStateVec);
}

__global__ void statevec_initClassicalStateKernel(long long int stateVecSize, Complex *stateVec, long long int stateInd){
    long long int index;

    // initialise the state to |stateInd>
    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;
    stateVec[index].real = 0.0;
    stateVec[index].imag = 0.0;

    if (index==stateInd){
        // classical state has probability 1
        stateVec[stateInd].real = 1.0;
        stateVec[stateInd].imag = 0.0;
    }
}

void statevec_initClassicalState(Qureg qureg, long long int stateInd)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_initClassicalStateKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        qureg.numAmpsPerChunk,
        qureg.deviceStateVec, stateInd);
}

__global__ void statevec_initStateDebugKernel(long long int stateVecSize, Complex *stateVec){
    long long int index;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    stateVec[index].real = (index*2.0)/10.0;
    stateVec[index].imag = (index*2.0+1.0)/10.0;
}

void statevec_initStateDebug(Qureg qureg)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_initStateDebugKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        qureg.numAmpsPerChunk,
        qureg.deviceStateVec);
}

__global__ void statevec_initStateOfSingleQubitKernel(long long int stateVecSize, Complex *stateVec, int qubitId, int outcome){
    long long int index;
    int bit;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    qreal normFactor = 1.0/sqrt((qreal)stateVecSize/2);
    bit = extractBit(qubitId, index);
    if (bit==outcome) {
        stateVec[index].real = normFactor;
        stateVec[index].imag = 0.0;
    } else {
        stateVec[index].real = 0.0;
        stateVec[index].imag = 0.0;
    }
}

void statevec_initStateOfSingleQubit(Qureg *qureg, int qubitId, int outcome)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg->numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_initStateOfSingleQubitKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg->numAmpsPerChunk,
                                                                               qureg->deviceStateVec,
                                                                               qubitId, outcome);
}

// returns 1 if successful, else 0
int statevec_initStateFromSingleFile(Qureg *qureg, char filename[200], QuESTEnv env){
    long long int chunkSize, stateVecSize;
    long long int indexInChunk, totalIndex;

    chunkSize = qureg->numAmpsPerChunk;
    stateVecSize = chunkSize*qureg->numChunks;


    FILE *fp;
    char line[200];

    fp = fopen(filename, "r");
    if (fp == NULL)
        return 0;

    indexInChunk = 0; totalIndex = 0;
    while (fgets(line, sizeof(char)*200, fp) != NULL && totalIndex<stateVecSize){
        if (line[0]!='#'){
            int chunkId = totalIndex/chunkSize;
            if (chunkId==qureg->chunkId){
                # if QuEST_PREC==1
                    sscanf(line, "%f, %f", &(qureg.stateVec[indexInChunk].real),
                            &(qureg.stateVec[indexInChunk].imag));
                # elif QuEST_PREC==2
                    sscanf(line, "%lf, %lf", &(qureg.stateVec[indexInChunk].real),
                            &(qureg.stateVec[indexInChunk].imag));
                # elif QuEST_PREC==4
                    sscanf(line, "%lf, %lf", &(qureg.stateVec[indexInChunk].real),
                            &(qureg.stateVec[indexInChunk].imag));
                # endif
                indexInChunk += 1;
            }
            totalIndex += 1;
        }
    }
    fclose(fp);
    copyStateToGPU(*qureg);

    // indicate success
    return 1;
}

int statevec_compareStates(Qureg mq1, Qureg mq2, qreal precision){
    qreal diff;
    int chunkSize = mq1.numAmpsPerChunk;

    copyStateFromGPU(mq1);
    copyStateFromGPU(mq2);

    for (int i=0; i<chunkSize; i++){
        diff = mq1.stateVec[i].real - mq2.stateVec[i].real;
        if (diff<0) diff *= -1;
        if (diff>precision) return 0;
        diff = mq1.stateVec[i].imag - mq2.stateVec[i].imag;
        if (diff<0) diff *= -1;
        if (diff>precision) return 0;
    }
    return 1;
}

__global__ void statevec_compactUnitaryKernel (Qureg qureg, const int rotQubit, Complex alpha, Complex beta){
    // ----- sizes
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                           // current block
         indexUp,indexLo;                                     // current index and corresponding index in lower half block

    // ----- temp variables
    qreal   stateRealUp,stateRealLo,                             // storage for previous state values
           stateImagUp,stateImagLo;                             // (used in updates)
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    sizeHalfBlock = 1LL << rotQubit;                               // size of blocks halved
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks

    // ---------------------------------------------------------------- //
    //            rotate                                                //
    // ---------------------------------------------------------------- //

    //! fix -- no necessary for GPU version
    qreal alphaImag=alpha.imag, alphaReal=alpha.real;
    qreal betaImag=beta.imag, betaReal=beta.real;

    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;

    thisBlock   = thisTask / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    // store current state vector values in temp variables
    stateRealUp = qureg.deviceStateVec[indexUp].real;
    stateImagUp = qureg.deviceStateVec[indexUp].imag;

    stateRealLo = qureg.deviceStateVec[indexLo].real;
    stateImagLo = qureg.deviceStateVec[indexLo].imag;

    // state[indexUp] = alpha * state[indexUp] - conj(beta)  * state[indexLo]
    qureg.deviceStateVec[indexUp].real = alphaReal*stateRealUp - alphaImag*stateImagUp
        - betaReal*stateRealLo - betaImag*stateImagLo;
    qureg.deviceStateVec[indexUp].imag = alphaReal*stateImagUp + alphaImag*stateRealUp
        - betaReal*stateImagLo + betaImag*stateRealLo;

    // state[indexLo] = beta  * state[indexUp] + conj(alpha) * state[indexLo]
    qureg.deviceStateVec[indexLo].real = betaReal*stateRealUp - betaImag*stateImagUp
        + alphaReal*stateRealLo + alphaImag*stateImagLo;
    qureg.deviceStateVec[indexLo].imag = betaReal*stateImagUp + betaImag*stateRealUp
        + alphaReal*stateImagLo - alphaImag*stateRealLo;
}

void statevec_compactUnitary(Qureg qureg, const int targetQubit, Complex alpha, Complex beta)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_compactUnitaryKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, targetQubit, alpha, beta);
}

__global__ void statevec_controlledCompactUnitaryKernel (Qureg qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta){
    // ----- sizes
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                           // current block
         indexUp,indexLo;                                     // current index and corresponding index in lower half block

    // ----- temp variables
    qreal   stateRealUp,stateRealLo,                             // storage for previous state values
           stateImagUp,stateImagLo;                             // (used in updates)
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    const long long int numTasks=qureg.numAmpsPerChunk>>1;
    int controlBit;

    sizeHalfBlock = 1LL << targetQubit;                               // size of blocks halved
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks

    // ---------------------------------------------------------------- //
    //            rotate                                                //
    // ---------------------------------------------------------------- //

    //! fix -- no necessary for GPU version
    qreal alphaImag=alpha.imag, alphaReal=alpha.real;
    qreal betaImag=beta.imag, betaReal=beta.real;

    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;

    thisBlock   = thisTask / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    controlBit = extractBit(controlQubit, indexUp);
    if (controlBit){
        // store current state vector values in temp variables
        stateRealUp = qureg.deviceStateVec[indexUp].real;
        stateImagUp = qureg.deviceStateVec[indexUp].imag;

        stateRealLo = qureg.deviceStateVec[indexLo].real;
        stateImagLo = qureg.deviceStateVec[indexLo].imag;

        // state[indexUp] = alpha * state[indexUp] - conj(beta)  * state[indexLo]
        qureg.deviceStateVec[indexUp].real = alphaReal*stateRealUp - alphaImag*stateImagUp
            - betaReal*stateRealLo - betaImag*stateImagLo;
        qureg.deviceStateVec[indexUp].imag = alphaReal*stateImagUp + alphaImag*stateRealUp
            - betaReal*stateImagLo + betaImag*stateRealLo;

        // state[indexLo] = beta  * state[indexUp] + conj(alpha) * state[indexLo]
        qureg.deviceStateVec[indexLo].real = betaReal*stateRealUp - betaImag*stateImagUp
            + alphaReal*stateRealLo + alphaImag*stateImagLo;
        qureg.deviceStateVec[indexLo].imag = betaReal*stateImagUp + betaImag*stateRealUp
            + alphaReal*stateImagLo - alphaImag*stateRealLo;
    }
}

void statevec_controlledCompactUnitary(Qureg qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_controlledCompactUnitaryKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, controlQubit, targetQubit, alpha, beta);
}

__global__ void statevec_unitaryKernel(Qureg qureg, const int targetQubit, ComplexMatrix2 u){
    // ----- sizes
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                           // current block
         indexUp,indexLo;                                     // current index and corresponding index in lower half block

    // ----- temp variables
    qreal   stateRealUp,stateRealLo,                             // storage for previous state values
           stateImagUp,stateImagLo;                             // (used in updates)
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    sizeHalfBlock = 1LL << targetQubit;                               // size of blocks halved
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks

    // ---------------------------------------------------------------- //
    //            rotate                                                //
    // ---------------------------------------------------------------- //

    //! fix -- no necessary for GPU version

    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;

    thisBlock   = thisTask / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    // store current state vector values in temp variables
    stateRealUp = qureg.deviceStateVec[indexUp].real;
    stateImagUp = qureg.deviceStateVec[indexUp].imag;

    stateRealLo = qureg.deviceStateVec[indexLo].real;
    stateImagLo = qureg.deviceStateVec[indexLo].imag;

    // state[indexUp] = u00 * state[indexUp] + u01 * state[indexLo]
    qureg.deviceStateVec[indexUp].real = u.r0c0.real*stateRealUp - u.r0c0.imag*stateImagUp
        + u.r0c1.real*stateRealLo - u.r0c1.imag*stateImagLo;
    qureg.deviceStateVec[indexUp].imag = u.r0c0.real*stateImagUp + u.r0c0.imag*stateRealUp
        + u.r0c1.real*stateImagLo + u.r0c1.imag*stateRealLo;

    // state[indexLo] = u10  * state[indexUp] + u11 * state[indexLo]
    qureg.deviceStateVec[indexLo].real = u.r1c0.real*stateRealUp  - u.r1c0.imag*stateImagUp
        + u.r1c1.real*stateRealLo  -  u.r1c1.imag*stateImagLo;
    qureg.deviceStateVec[indexLo].imag = u.r1c0.real*stateImagUp + u.r1c0.imag*stateRealUp
        + u.r1c1.real*stateImagLo + u.r1c1.imag*stateRealLo;
}

void statevec_unitary(Qureg qureg, const int targetQubit, ComplexMatrix2 u)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_unitaryKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, targetQubit, u);
}

__global__ void statevec_controlledUnitaryKernel(Qureg qureg, const int controlQubit, const int targetQubit, ComplexMatrix2 u){
    // ----- sizes
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                           // current block
         indexUp,indexLo;                                     // current index and corresponding index in lower half block

    // ----- temp variables
    qreal   stateRealUp,stateRealLo,                             // storage for previous state values
           stateImagUp,stateImagLo;                             // (used in updates)
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    int controlBit;

    sizeHalfBlock = 1LL << targetQubit;                               // size of blocks halved
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks

    // ---------------------------------------------------------------- //
    //            rotate                                                //
    // ---------------------------------------------------------------- //

    //! fix -- no necessary for GPU version

    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;

    thisBlock   = thisTask / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    // store current state vector values in temp variables
    stateRealUp = qureg.deviceStateVec[indexUp].real;
    stateImagUp = qureg.deviceStateVec[indexUp].imag;

    stateRealLo = qureg.deviceStateVec[indexLo].real;
    stateImagLo = qureg.deviceStateVec[indexLo].imag;

    controlBit = extractBit(controlQubit, indexUp);
    if (controlBit){
        // state[indexUp] = u00 * state[indexUp] + u01 * state[indexLo]
        qureg.deviceStateVec[indexUp].real = u.r0c0.real*stateRealUp - u.r0c0.imag*stateImagUp
            + u.r0c1.real*stateRealLo - u.r0c1.imag*stateImagLo;
        qureg.deviceStateVec[indexUp].imag = u.r0c0.real*stateImagUp + u.r0c0.imag*stateRealUp
            + u.r0c1.real*stateImagLo + u.r0c1.imag*stateRealLo;

        // state[indexLo] = u10  * state[indexUp] + u11 * state[indexLo]
        qureg.deviceStateVec[indexLo].real = u.r1c0.real*stateRealUp  - u.r1c0.imag*stateImagUp
            + u.r1c1.real*stateRealLo  -  u.r1c1.imag*stateImagLo;
        qureg.deviceStateVec[indexLo].imag = u.r1c0.real*stateImagUp + u.r1c0.imag*stateRealUp
            + u.r1c1.real*stateImagLo + u.r1c1.imag*stateRealLo;
    }
}

void statevec_controlledUnitary(Qureg qureg, const int controlQubit, const int targetQubit, ComplexMatrix2 u)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_controlledUnitaryKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, controlQubit, targetQubit, u);
}

__global__ void statevec_multiControlledUnitaryKernel(Qureg qureg, long long int mask, const int targetQubit, ComplexMatrix2 u){
    // ----- sizes
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                           // current block
         indexUp,indexLo;                                     // current index and corresponding index in lower half block

    // ----- temp variables
    qreal   stateRealUp,stateRealLo,                             // storage for previous state values
           stateImagUp,stateImagLo;                             // (used in updates)
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    const long long int numTasks=qureg.numAmpsPerChunk>>1;


    sizeHalfBlock = 1LL << targetQubit;                               // size of blocks halved
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks

    // ---------------------------------------------------------------- //
    //            rotate                                                //
    // ---------------------------------------------------------------- //

    //! fix -- no necessary for GPU version

    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;

    thisBlock   = thisTask / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    if (mask == (mask & indexUp) ){
        // store current state vector values in temp variables
        stateRealUp = qureg.deviceStateVec[indexUp].real;
        stateImagUp = qureg.deviceStateVec[indexUp].imag;

        stateRealLo = qureg.deviceStateVec[indexLo].real;
        stateImagLo = qureg.deviceStateVec[indexLo].imag;

        // state[indexUp] = u00 * state[indexUp] + u01 * state[indexLo]
        qureg.deviceStateVec[indexUp].real = u.r0c0.real*stateRealUp - u.r0c0.imag*stateImagUp
            + u.r0c1.real*stateRealLo - u.r0c1.imag*stateImagLo;
        qureg.deviceStateVec[indexUp].imag = u.r0c0.real*stateImagUp + u.r0c0.imag*stateRealUp
            + u.r0c1.real*stateImagLo + u.r0c1.imag*stateRealLo;

        // state[indexLo] = u10  * state[indexUp] + u11 * state[indexLo]
        qureg.deviceStateVec[indexLo].real = u.r1c0.real*stateRealUp  - u.r1c0.imag*stateImagUp
            + u.r1c1.real*stateRealLo  -  u.r1c1.imag*stateImagLo;
        qureg.deviceStateVec[indexLo].imag = u.r1c0.real*stateImagUp + u.r1c0.imag*stateRealUp
            + u.r1c1.real*stateImagLo + u.r1c1.imag*stateRealLo;
    }
}

void statevec_multiControlledUnitary(Qureg qureg, int *controlQubits, int numControlQubits, const int targetQubit, ComplexMatrix2 u)
{
    int threadsPerCUDABlock, CUDABlocks;
    long long int mask=0;
    for (int i=0; i<numControlQubits; i++) mask = mask | (1LL<<controlQubits[i]);
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_multiControlledUnitaryKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, mask, targetQubit, u);
}

__global__ void statevec_pauliXKernel(Qureg qureg, const int targetQubit){
    // ----- sizes
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                           // current block
         indexUp,indexLo;                                     // current index and corresponding index in lower half block

    // ----- temp variables
    qreal   stateRealUp,                             // storage for previous state values
           stateImagUp;                             // (used in updates)
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    sizeHalfBlock = 1LL << targetQubit;                               // size of blocks halved
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks

    // ---------------------------------------------------------------- //
    //            rotate                                                //
    // ---------------------------------------------------------------- //

    //! fix -- no necessary for GPU version

    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;

    thisBlock   = thisTask / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    // store current state vector values in temp variables
    stateRealUp = qureg.deviceStateVec[indexUp].real;
    stateImagUp = qureg.deviceStateVec[indexUp].imag;

    qureg.deviceStateVec[indexUp].real = qureg.deviceStateVec[indexLo].real;
    qureg.deviceStateVec[indexUp].imag = qureg.deviceStateVec[indexLo].imag;

    qureg.deviceStateVec[indexLo].real = stateRealUp;
    qureg.deviceStateVec[indexLo].imag = stateImagUp;
}

void statevec_pauliX(Qureg qureg, const int targetQubit)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_pauliXKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, targetQubit);
}

__global__ void statevec_pauliYKernel(Qureg qureg, const int targetQubit, const int conjFac){

    long long int sizeHalfBlock = 1LL << targetQubit;
    long long int sizeBlock     = 2LL * sizeHalfBlock;
    long long int numTasks      = qureg.numAmpsPerChunk >> 1;
    long long int thisTask      = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;

    long long int thisBlock     = thisTask / sizeHalfBlock;
    long long int indexUp       = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    long long int indexLo       = indexUp + sizeHalfBlock;
    qreal  stateRealUp, stateImagUp;

    stateRealUp = qureg.deviceStateVec[indexUp].real;
    stateImagUp = qureg.deviceStateVec[indexUp].imag;

    // update under +-{{0, -i}, {i, 0}}
    qureg.deviceStateVec[indexUp].real = conjFac * qureg.deviceStateVec[indexLo].imag;
    qureg.deviceStateVec[indexUp].imag = conjFac * -qureg.deviceStateVec[indexLo].real;
    qureg.deviceStateVec[indexLo].real = conjFac * -stateImagUp;
    qureg.deviceStateVec[indexLo].imag = conjFac * stateRealUp;
}

void statevec_pauliY(Qureg qureg, const int targetQubit)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_pauliYKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, targetQubit, 1);
}

void statevec_pauliYConj(Qureg qureg, const int targetQubit)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_pauliYKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, targetQubit, -1);
}

__global__ void statevec_controlledPauliYKernel(Qureg qureg, const int controlQubit, const int targetQubit, const int conjFac)
{
    long long int index;
    long long int sizeBlock, sizeHalfBlock;
    long long int stateVecSize;
    int controlBit;

    qreal   stateRealUp, stateImagUp;
    long long int thisBlock, indexUp, indexLo;
    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;

    stateVecSize = qureg.numAmpsPerChunk;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=(stateVecSize>>1)) return;
    thisBlock   = index / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + index%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    controlBit = extractBit(controlQubit, indexUp);
    if (controlBit){

        stateRealUp = qureg.deviceStateVec[indexUp].real;
        stateImagUp = qureg.deviceStateVec[indexUp].imag;

        // update under +-{{0, -i}, {i, 0}}
        qureg.deviceStateVec[indexUp].real = conjFac * qureg.deviceStateVec[indexLo].imag;
        qureg.deviceStateVec[indexUp].imag = conjFac * -qureg.deviceStateVec[indexLo].real;
        qureg.deviceStateVec[indexLo].real = conjFac * -stateImagUp;
        qureg.deviceStateVec[indexLo].imag = conjFac * stateRealUp;
    }
}

void statevec_controlledPauliY(Qureg qureg, const int controlQubit, const int targetQubit)
{
    int conjFactor = 1;
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_controlledPauliYKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, controlQubit, targetQubit, conjFactor);
}

void statevec_controlledPauliYConj(Qureg qureg, const int controlQubit, const int targetQubit)
{
    int conjFactor = -1;
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_controlledPauliYKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, controlQubit, targetQubit, conjFactor);
}

__global__ void statevec_phaseShiftByTermKernel(Qureg qureg, const int targetQubit, qreal cosAngle, qreal sinAngle) {

    long long int sizeBlock, sizeHalfBlock;
    long long int thisBlock, indexUp,indexLo;

    qreal stateRealLo, stateImagLo;
    long long int thisTask;
    const long long int numTasks = qureg.numAmpsPerChunk >> 1;

    sizeHalfBlock = 1LL << targetQubit;
    sizeBlock     = 2LL * sizeHalfBlock;


    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;
    thisBlock   = thisTask / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    stateRealLo = qureg.deviceStateVec[indexLo].real;
    stateImagLo = qureg.deviceStateVec[indexLo].imag;

    qureg.deviceStateVec[indexLo].real = cosAngle*stateRealLo - sinAngle*stateImagLo;
    qureg.deviceStateVec[indexLo].imag = sinAngle*stateRealLo + cosAngle*stateImagLo;
}

void statevec_phaseShiftByTerm(Qureg qureg, const int targetQubit, Complex term)
{
    qreal cosAngle = term.real;
    qreal sinAngle = term.imag;

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_phaseShiftByTermKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, targetQubit, cosAngle, sinAngle);
}

__global__ void statevec_controlledPhaseShiftKernel(Qureg qureg, const int idQubit1, const int idQubit2, qreal cosAngle, qreal sinAngle)
{
    long long int index;
    long long int stateVecSize;
    int bit1, bit2;
    qreal stateRealLo, stateImagLo;

    stateVecSize = qureg.numAmpsPerChunk;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    bit1 = extractBit (idQubit1, index);
    bit2 = extractBit (idQubit2, index);
    if (bit1 && bit2) {
        stateRealLo = qureg.deviceStateVec[index].real;
        stateImagLo = qureg.deviceStateVec[index].imag;

        qureg.deviceStateVec[index].real = cosAngle*stateRealLo - sinAngle*stateImagLo;
        qureg.deviceStateVec[index].imag = sinAngle*stateRealLo + cosAngle*stateImagLo;
    }
}

void statevec_controlledPhaseShift(Qureg qureg, const int idQubit1, const int idQubit2, qreal angle)
{
    qreal cosAngle = cos(angle);
    qreal sinAngle = sin(angle);

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_controlledPhaseShiftKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, idQubit1, idQubit2, cosAngle, sinAngle);
}

__global__ void statevec_multiControlledPhaseShiftKernel(Qureg qureg, long long int mask, qreal cosAngle, qreal sinAngle) {
    qreal stateRealLo, stateImagLo;
    long long int index;
    long long int stateVecSize;

    stateVecSize = qureg.numAmpsPerChunk;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    if (mask == (mask & index) ){
        stateRealLo = qureg.deviceStateVec[index].real;
        stateImagLo = qureg.deviceStateVec[index].imag;
        qureg.deviceStateVec[index].real = cosAngle*stateRealLo - sinAngle*stateImagLo;
        qureg.deviceStateVec[index].imag = sinAngle*stateRealLo + cosAngle*stateImagLo;
    }
}

void statevec_multiControlledPhaseShift(Qureg qureg, int *controlQubits, int numControlQubits, qreal angle)
{
    qreal cosAngle = cos(angle);
    qreal sinAngle = sin(angle);

    long long int mask=0;
    for (int i=0; i<numControlQubits; i++)
        mask = mask | (1LL<<controlQubits[i]);

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_multiControlledPhaseShiftKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, mask, cosAngle, sinAngle);
}

qreal densmatr_calcTotalProb(Qureg qureg) {

    // computes the trace using Kahan summation
    qreal pTotal=0;
    qreal y, t, c;
    c = 0;

    long long int numCols = 1LL << qureg.numQubitsRepresented;
    long long diagIndex;

    copyStateFromGPU(qureg);

    for (int col=0; col< numCols; col++) {
        diagIndex = col*(numCols + 1);
        y = qureg.stateVec[diagIndex].real - c;
        t = pTotal + y;
        c = ( t - pTotal ) - y; // brackets are important
        pTotal = t;
    }

    return pTotal;
}

qreal statevec_calcTotalProb(Qureg qureg){
    /* IJB - implemented using Kahan summation for greater accuracy at a slight floating
       point operation overhead. For more details see https://en.wikipedia.org/wiki/Kahan_summation_algorithm */
    /* Don't change the bracketing in this routine! */
    qreal pTotal=0;
    qreal y, t, c;
    long long int index;
    long long int numAmpsPerRank = qureg.numAmpsPerChunk;

    copyStateFromGPU(qureg);

    c = 0.0;
    for (index=0; index<numAmpsPerRank; index++){
        /* Perform pTotal+=qureg.stateVec[index].real*qureg.stateVec[index].real; by Kahan */
        // pTotal+=qureg.stateVec[index].real*qureg.stateVec[index].real;
        y = qureg.stateVec[index].real*qureg.stateVec[index].real - c;
        t = pTotal + y;
        c = ( t - pTotal ) - y;
        pTotal = t;

        /* Perform pTotal+=qureg.stateVec[index].imag*qureg.stateVec[index].imag; by Kahan */
        //pTotal+=qureg.stateVec[index].imag*qureg.stateVec[index].imag;
        y = qureg.stateVec[index].imag*qureg.stateVec[index].imag - c;
        t = pTotal + y;
        c = ( t - pTotal ) - y;
        pTotal = t;


    }
    return pTotal;
}

__global__ void statevec_controlledPhaseFlipKernel(Qureg qureg, const int idQubit1, const int idQubit2)
{
    long long int index;
    long long int stateVecSize;
    int bit1, bit2;

    stateVecSize = qureg.numAmpsPerChunk;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    bit1 = extractBit (idQubit1, index);
    bit2 = extractBit (idQubit2, index);
    if (bit1 && bit2) {
        qureg.deviceStateVec[index].real = - qureg.deviceStateVec[index].real;
        qureg.deviceStateVec[index].imag = - qureg.deviceStateVec[index].imag;
    }
}

void statevec_controlledPhaseFlip(Qureg qureg, const int idQubit1, const int idQubit2)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_controlledPhaseFlipKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, idQubit1, idQubit2);
}

__global__ void statevec_multiControlledPhaseFlipKernel(Qureg qureg, long long int mask)
{
    long long int index;
    long long int stateVecSize;

    stateVecSize = qureg.numAmpsPerChunk;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=stateVecSize) return;

    if (mask == (mask & index) ){
        qureg.deviceStateVec[index].real = - qureg.deviceStateVec[index].real;
        qureg.deviceStateVec[index].imag = - qureg.deviceStateVec[index].imag;
    }
}

void statevec_multiControlledPhaseFlip(Qureg qureg, int *controlQubits, int numControlQubits)
{
    int threadsPerCUDABlock, CUDABlocks;
    long long int mask=0;
    for (int i=0; i<numControlQubits; i++) mask = mask | (1LL<<controlQubits[i]);
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_multiControlledPhaseFlipKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, mask);
}


__global__ void statevec_hadamardKernel (Qureg qureg, const int targetQubit){
    // ----- sizes
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                           // current block
         indexUp,indexLo;                                     // current index and corresponding index in lower half block

    // ----- temp variables
    qreal   stateRealUp,stateRealLo,                             // storage for previous state values
           stateImagUp,stateImagLo;                             // (used in updates)
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    const long long int numTasks=qureg.numAmpsPerChunk>>1;

    sizeHalfBlock = 1LL << targetQubit;                               // size of blocks halved
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks

    // ---------------------------------------------------------------- //
    //            rotate                                                //
    // ---------------------------------------------------------------- //

    //! fix -- no necessary for GPU version

    qreal recRoot2 = 1.0/sqrt(2.0);

    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;

    thisBlock   = thisTask / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    // store current state vector values in temp variables
    stateRealUp = qureg.deviceStateVec[indexUp].real;
    stateImagUp = qureg.deviceStateVec[indexUp].imag;

    stateRealLo = qureg.deviceStateVec[indexLo].real;
    stateImagLo = qureg.deviceStateVec[indexLo].imag;

    qureg.deviceStateVec[indexUp].real = recRoot2*(stateRealUp + stateRealLo);
    qureg.deviceStateVec[indexUp].imag = recRoot2*(stateImagUp + stateImagLo);

    qureg.deviceStateVec[indexLo].real = recRoot2*(stateRealUp - stateRealLo);
    qureg.deviceStateVec[indexLo].imag = recRoot2*(stateImagUp - stateImagLo);
}

void statevec_hadamard(Qureg qureg, const int targetQubit)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_hadamardKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, targetQubit);
}

__global__ void statevec_controlledNotKernel(Qureg qureg, const int controlQubit, const int targetQubit)
{
    long long int index;
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    long long int stateVecSize;
    int controlBit;

    // ----- temp variables
    qreal   stateRealUp,                             // storage for previous state values
           stateImagUp;                             // (used in updates)
    long long int thisBlock,                                           // current block
         indexUp,indexLo;                                     // current index and corresponding index in lower half block
    sizeHalfBlock = 1LL << targetQubit;                               // size of blocks halved
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks

    stateVecSize = qureg.numAmpsPerChunk;

    index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index>=(stateVecSize>>1)) return;
    thisBlock   = index / sizeHalfBlock;
    indexUp     = thisBlock*sizeBlock + index%sizeHalfBlock;
    indexLo     = indexUp + sizeHalfBlock;

    controlBit = extractBit(controlQubit, indexUp);
    if (controlBit){
        stateRealUp = qureg.deviceStateVec[indexUp].real;
        stateImagUp = qureg.deviceStateVec[indexUp].imag;

        qureg.deviceStateVec[indexUp].real = qureg.deviceStateVec[indexLo].real;
        qureg.deviceStateVec[indexUp].imag = qureg.deviceStateVec[indexLo].imag;

        qureg.deviceStateVec[indexLo].real = stateRealUp;
        qureg.deviceStateVec[indexLo].imag = stateImagUp;
    }
}

void statevec_controlledNot(Qureg qureg, const int controlQubit, const int targetQubit)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk)/threadsPerCUDABlock);
    statevec_controlledNotKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, controlQubit, targetQubit);
}

__device__ __host__ unsigned int log2Int( unsigned int x )
{
    unsigned int ans = 0 ;
    while( x>>=1 ) ans++;
    return ans ;
}

__device__ void reduceBlock(qreal *arrayIn, qreal *reducedArray, int length){
    int i, l, r;
    int threadMax, maxDepth;
    threadMax = length/2;
    maxDepth = log2Int(length/2);

    for (i=0; i<maxDepth+1; i++){
        if (threadIdx.x<threadMax){
            l = threadIdx.x;
            r = l + threadMax;
            arrayIn[l] = arrayIn[r] + arrayIn[l];
        }
        threadMax = threadMax >> 1;
        __syncthreads(); // optimise -- use warp shuffle instead
    }

    if (threadIdx.x==0) reducedArray[blockIdx.x] = arrayIn[0];
}

__global__ void copySharedReduceBlock(qreal*arrayIn, qreal *reducedArray, int length){
    extern __shared__ qreal tempReductionArray[];
    int blockOffset = blockIdx.x*length;
    tempReductionArray[threadIdx.x*2] = arrayIn[blockOffset + threadIdx.x*2];
    tempReductionArray[threadIdx.x*2+1] = arrayIn[blockOffset + threadIdx.x*2+1];
    __syncthreads();
    reduceBlock(tempReductionArray, reducedArray, length);
}

__global__ void densmatr_findProbabilityOfZeroKernel(
    Qureg qureg, const int measureQubit, qreal *reducedArray
) {
    // run by each thread
    // use of block here refers to contiguous amplitudes where measureQubit = 0,
    // (then =1) and NOT the CUDA block, which is the partitioning of CUDA threads

    long long int densityDim    = 1LL << qureg.numQubitsRepresented;
    long long int numTasks      = densityDim >> 1;
    long long int sizeHalfBlock = 1LL << (measureQubit);
    long long int sizeBlock     = 2LL * sizeHalfBlock;

    long long int thisBlock;    // which block this thread is processing
    long long int thisTask;     // which part of the block this thread is processing
    long long int basisIndex;   // index of this thread's computational basis state
    long long int densityIndex; // " " index of |basis><basis| in the flat density matrix

    // array of each thread's collected probability, to be summed
    extern __shared__ qreal tempReductionArray[];

    // figure out which density matrix prob that this thread is assigned
    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;
    thisBlock = thisTask / sizeHalfBlock;
    basisIndex = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    densityIndex = (densityDim + 1) * basisIndex;

    // record the probability in the CUDA-BLOCK-wide array
    qreal prob = qureg.deviceStateVec[densityIndex].real;   // im[densityIndex] assumed ~ 0
    tempReductionArray[threadIdx.x] = prob;

    // sum the probs collected by this CUDA-BLOCK's threads into a per-CUDA-BLOCK array
    __syncthreads();
    if (threadIdx.x<blockDim.x/2){
        reduceBlock(tempReductionArray, reducedArray, blockDim.x);
    }
}

__global__ void statevec_findProbabilityOfZeroKernel(
        Qureg qureg, const int measureQubit, qreal *reducedArray
) {
    // ----- sizes
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                           // current block
         index;                                               // current index for first half block
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    long long int numTasks=qureg.numAmpsPerChunk>>1;
    // (good for shared memory parallelism)

    extern __shared__ qreal tempReductionArray[];

    // ---------------------------------------------------------------- //
    //            dimensions                                            //
    // ---------------------------------------------------------------- //
    sizeHalfBlock = 1LL << (measureQubit);                       // number of state vector elements to sum,
    // and then the number to skip
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks (pairs of measure and skip entries)

    // ---------------------------------------------------------------- //
    //            find probability                                      //
    // ---------------------------------------------------------------- //

    //
    // --- task-based shared-memory parallel implementation
    //


    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;

    thisBlock = thisTask / sizeHalfBlock;
    index     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;
    qreal realVal, imagVal;
    realVal = qureg.deviceStateVec[index].real;
    imagVal = qureg.deviceStateVec[index].imag;
    tempReductionArray[threadIdx.x] = realVal*realVal + imagVal*imagVal;
    __syncthreads();

    if (threadIdx.x<blockDim.x/2){
        reduceBlock(tempReductionArray, reducedArray, blockDim.x);
    }
}

int getNumReductionLevels(long long int numValuesToReduce, int numReducedPerLevel){
    int levels=0;
    while (numValuesToReduce){
        numValuesToReduce = numValuesToReduce/numReducedPerLevel;
        levels++;
    }
    return levels;
}

void swapDouble(qreal **a, qreal **b){
    qreal *temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

qreal densmatr_findProbabilityOfZero(Qureg qureg, const int measureQubit)
{
    long long int densityDim = 1LL << qureg.numQubitsRepresented;
    long long int numValuesToReduce = densityDim >> 1;  // half of the diagonal has measureQubit=0

    int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
    int maxReducedPerLevel = REDUCE_SHARED_SIZE;
    int firstTime = 1;

    while (numValuesToReduce > 1) {

        // need less than one CUDA-BLOCK to reduce
        if (numValuesToReduce < maxReducedPerLevel) {
            valuesPerCUDABlock = numValuesToReduce;
            numCUDABlocks = 1;
        }
        // otherwise use only full CUDA-BLOCKS
        else {
            valuesPerCUDABlock = maxReducedPerLevel; // constrained by shared memory
            numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
        }

        sharedMemSize = valuesPerCUDABlock*sizeof(qreal);

        // spawn threads to sum the probs in each block
        if (firstTime) {
            densmatr_findProbabilityOfZeroKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
                qureg, measureQubit, qureg.firstLevelReduction);
            firstTime = 0;

        // sum the block probs
        } else {
            cudaDeviceSynchronize();
            copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                    qureg.firstLevelReduction,
                    qureg.secondLevelReduction, valuesPerCUDABlock);
            cudaDeviceSynchronize();
            swapDouble(&(qureg.firstLevelReduction), &(qureg.secondLevelReduction));
        }

        numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
    }

    qreal zeroProb;
    cudaMemcpy(&zeroProb, qureg.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);
    return zeroProb;
}

qreal statevec_findProbabilityOfZero(Qureg qureg, const int measureQubit)
{
    long long int numValuesToReduce = qureg.numAmpsPerChunk>>1;
    int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
    qreal stateProb=0;
    int firstTime=1;
    int maxReducedPerLevel = REDUCE_SHARED_SIZE;

    while(numValuesToReduce>1){
        if (numValuesToReduce<maxReducedPerLevel){
            // Need less than one CUDA block to reduce values
            valuesPerCUDABlock = numValuesToReduce;
            numCUDABlocks = 1;
        } else {
            // Use full CUDA blocks, with block size constrained by shared mem usage
            valuesPerCUDABlock = maxReducedPerLevel;
            numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
        }
        sharedMemSize = valuesPerCUDABlock*sizeof(qreal);

        if (firstTime){
            statevec_findProbabilityOfZeroKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
                    qureg, measureQubit, qureg.firstLevelReduction);
            firstTime=0;
        } else {
            cudaDeviceSynchronize();
            copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                    qureg.firstLevelReduction,
                    qureg.secondLevelReduction, valuesPerCUDABlock);
            cudaDeviceSynchronize();
            swapDouble(&(qureg.firstLevelReduction), &(qureg.secondLevelReduction));
        }
        numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
    }
    cudaMemcpy(&stateProb, qureg.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);
    return stateProb;
}

qreal statevec_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome)
{
    qreal outcomeProb = statevec_findProbabilityOfZero(qureg, measureQubit);
    if (outcome==1)
        outcomeProb = 1.0 - outcomeProb;
    return outcomeProb;
}

qreal densmatr_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome)
{
    qreal outcomeProb = densmatr_findProbabilityOfZero(qureg, measureQubit);
    if (outcome==1)
        outcomeProb = 1.0 - outcomeProb;
    return outcomeProb;
}


/** computes either a real or imag term in the inner product */
__global__ void statevec_calcInnerProductKernel(
    int getRealComp,
    Complex* stateVec1, Complex* stateVec2
    long long int numTermsToSum, qreal* reducedArray)
{
    long long int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numTermsToSum) return;

    // choose whether to calculate the real or imaginary term of the inner product
    qreal innerProdTerm;
    if (getRealComp)
        innerProdTerm = stateVec1[index].real*stateVec2[index].real + stateVec1[index].imag*stateVec2[index].imag;
    else
        innerProdTerm = stateVec1[index].real*stateVec2[index].imag - stateVec1[index].imag*stateVec2[index].real;

    // array of each thread's collected probability, to be summed
    extern __shared__ qreal tempReductionArray[];
    tempReductionArray[threadIdx.x] = innerProdTerm;
    __syncthreads();

    // every second thread reduces
    if (threadIdx.x<blockDim.x/2)
        reduceBlock(tempReductionArray, reducedArray, blockDim.x);
}

/** Terrible code which unnecessarily individually computes and sums the real and imaginary components of the
 * inner product, so as to not have to worry about keeping the sums separated during reduction.
 * Truly disgusting, probably doubles runtime, please fix.
 * @TODO could even do the kernel twice, storing real in bra.reduc and imag in ket.reduc?
 */
Complex statevec_calcInnerProduct(Qureg bra, Qureg ket) {

    qreal innerProdReal, innerProdImag;

    int getRealComp;
    long long int numValuesToReduce;
    int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
    int maxReducedPerLevel;
    int firstTime;

    // compute real component of inner product
    getRealComp = 1;
    numValuesToReduce = bra.numAmpsPerChunk;
    maxReducedPerLevel = REDUCE_SHARED_SIZE;
    firstTime = 1;
    while (numValuesToReduce > 1) {
        if (numValuesToReduce < maxReducedPerLevel) {
            valuesPerCUDABlock = numValuesToReduce;
            numCUDABlocks = 1;
        }
        else {
            valuesPerCUDABlock = maxReducedPerLevel;
            numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
        }
        sharedMemSize = valuesPerCUDABlock*sizeof(qreal);
        if (firstTime) {
             statevec_calcInnerProductKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
                 getRealComp,
                 bra.deviceStateVec,
                 ket.deviceStateVec,
                 numValuesToReduce,
                 bra.firstLevelReduction);
            firstTime = 0;
        } else {
            cudaDeviceSynchronize();
            copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                    bra.firstLevelReduction,
                    bra.secondLevelReduction, valuesPerCUDABlock);
            cudaDeviceSynchronize();
            swapDouble(&(bra.firstLevelReduction), &(bra.secondLevelReduction));
        }
        numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
    }
    cudaMemcpy(&innerProdReal, bra.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);

    // compute imag component of inner product
    getRealComp = 0;
    numValuesToReduce = bra.numAmpsPerChunk;
    maxReducedPerLevel = REDUCE_SHARED_SIZE;
    firstTime = 1;
    while (numValuesToReduce > 1) {
        if (numValuesToReduce < maxReducedPerLevel) {
            valuesPerCUDABlock = numValuesToReduce;
            numCUDABlocks = 1;
        }
        else {
            valuesPerCUDABlock = maxReducedPerLevel;
            numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
        }
        sharedMemSize = valuesPerCUDABlock*sizeof(qreal);
        if (firstTime) {
             statevec_calcInnerProductKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
                 getRealComp,
                 bra.deviceStateVec,
                 ket.deviceStateVec,
                 numValuesToReduce,
                 bra.firstLevelReduction);
            firstTime = 0;
        } else {
            cudaDeviceSynchronize();
            copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                    bra.firstLevelReduction,
                    bra.secondLevelReduction, valuesPerCUDABlock);
            cudaDeviceSynchronize();
            swapDouble(&(bra.firstLevelReduction), &(bra.secondLevelReduction));
        }
        numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
    }
    cudaMemcpy(&innerProdImag, bra.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);

    // return complex
    Complex innerProd;
    innerProd.real = innerProdReal;
    innerProd.imag = innerProdImag;
    return innerProd;
}

/** computes one term of (vec^*T) dens * vec */
__global__ void densmatr_calcFidelityKernel(Qureg dens, Qureg vec, long long int dim, qreal* reducedArray) {

    // figure out which density matrix row to consider
    long long int col;
    long long int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= dim) return;

    // compute the row-th element of the product dens*vec
    qreal prodReal = 0;
    qreal prodImag = 0;
    for (col=0LL; col < dim; col++) {
        qreal densElemReal = dens.deviceStateVec[dim*col + row].real;
        qreal densElemImag = dens.deviceStateVec[dim*col + row].imag;

        prodReal += densElemReal*vec.deviceStateVec[col].real - densElemImag*vec.deviceStateVec[col].imag;
        prodImag += densElemReal*vec.deviceStateVec[col].imag + densElemImag*vec.deviceStateVec[col].real;
    }

    // multiply with row-th elem of (vec^*)
    qreal termReal = prodImag*vec[row].imag + prodReal*vec[row].real;

    // imag of every term should be zero, because each is a valid fidelity calc of an eigenstate
    //qreal termImag = prodImag*vec[row].real - prodReal*vec[row].imag;

    extern __shared__ qreal tempReductionArray[];
    tempReductionArray[threadIdx.x] = termReal;
    __syncthreads();

    // every second thread reduces
    if (threadIdx.x<blockDim.x/2)
        reduceBlock(tempReductionArray, reducedArray, blockDim.x);
}

// @TODO implement
qreal densmatr_calcFidelity(Qureg qureg, Qureg pureState) {

    // we're summing the square of every term in the density matrix
    long long int densityDim = 1LL << qureg.numQubitsRepresented;
    long long int numValuesToReduce = densityDim;

    int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
    int maxReducedPerLevel = REDUCE_SHARED_SIZE;
    int firstTime = 1;

    while (numValuesToReduce > 1) {

        // need less than one CUDA-BLOCK to reduce
        if (numValuesToReduce < maxReducedPerLevel) {
            valuesPerCUDABlock = numValuesToReduce;
            numCUDABlocks = 1;
        }
        // otherwise use only full CUDA-BLOCKS
        else {
            valuesPerCUDABlock = maxReducedPerLevel; // constrained by shared memory
            numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
        }
        // dictates size of reduction array
        sharedMemSize = valuesPerCUDABlock*sizeof(qreal);

        // spawn threads to sum the probs in each block
        // store the reduction in the pureState array
        if (firstTime) {
             densmatr_calcFidelityKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
                 qureg, pureState, densityDim, pureState.firstLevelReduction);
            firstTime = 0;

        // sum the block probs
        } else {
            cudaDeviceSynchronize();
            copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                    pureState.firstLevelReduction,
                    pureState.secondLevelReduction, valuesPerCUDABlock);
            cudaDeviceSynchronize();
            swapDouble(&(pureState.firstLevelReduction), &(pureState.secondLevelReduction));
        }

        numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
    }

    qreal fidelity;
    cudaMemcpy(&fidelity, pureState.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);
    return fidelity;
}


__global__ void densmatr_calcPurityKernel(Complex* vec, long long int numAmpsToSum, qreal *reducedArray) {

    // figure out which density matrix term this thread is assigned
    long long int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= numAmpsToSum) return;

    qreal term = vec[index].real*vec[index].real + vec[index].imag*vec[index].imag;

    // array of each thread's collected probability, to be summed
    extern __shared__ qreal tempReductionArray[];
    tempReductionArray[threadIdx.x] = term;
    __syncthreads();

    // every second thread reduces
    if (threadIdx.x<blockDim.x/2)
        reduceBlock(tempReductionArray, reducedArray, blockDim.x);
}

/** Computes the trace of the density matrix squared */
qreal densmatr_calcPurity(Qureg qureg) {

    // we're summing the square of every term in the density matrix
    long long int numValuesToReduce = qureg.numAmpsPerChunk;

    int valuesPerCUDABlock, numCUDABlocks, sharedMemSize;
    int maxReducedPerLevel = REDUCE_SHARED_SIZE;
    int firstTime = 1;

    while (numValuesToReduce > 1) {

        // need less than one CUDA-BLOCK to reduce
        if (numValuesToReduce < maxReducedPerLevel) {
            valuesPerCUDABlock = numValuesToReduce;
            numCUDABlocks = 1;
        }
        // otherwise use only full CUDA-BLOCKS
        else {
            valuesPerCUDABlock = maxReducedPerLevel; // constrained by shared memory
            numCUDABlocks = ceil((qreal)numValuesToReduce/valuesPerCUDABlock);
        }
        // dictates size of reduction array
        sharedMemSize = valuesPerCUDABlock*sizeof(qreal);

        // spawn threads to sum the probs in each block
        if (firstTime) {
             densmatr_calcPurityKernel<<<numCUDABlocks, valuesPerCUDABlock, sharedMemSize>>>(
                 qureg.deviceStateVec.real, qureg.deviceStateVec.imag,
                 numValuesToReduce, qureg.firstLevelReduction);
            firstTime = 0;

        // sum the block probs
        } else {
            cudaDeviceSynchronize();
            copySharedReduceBlock<<<numCUDABlocks, valuesPerCUDABlock/2, sharedMemSize>>>(
                    qureg.firstLevelReduction,
                    qureg.secondLevelReduction, valuesPerCUDABlock);
            cudaDeviceSynchronize();
            swapDouble(&(qureg.firstLevelReduction), &(qureg.secondLevelReduction));
        }

        numValuesToReduce = numValuesToReduce/maxReducedPerLevel;
    }

    qreal traceDensSquared;
    cudaMemcpy(&traceDensSquared, qureg.firstLevelReduction, sizeof(qreal), cudaMemcpyDeviceToHost);
    return traceDensSquared;
}

__global__ void statevec_collapseToKnownProbOutcomeKernel(Qureg qureg, int measureQubit, int outcome, qreal totalProbability)
{
    // ----- sizes
    long long int sizeBlock,                                           // size of blocks
         sizeHalfBlock;                                       // size of blocks halved
    // ----- indices
    long long int thisBlock,                                           // current block
         index;                                               // current index for first half block
    // ----- measured probability
    qreal   renorm;                                    // probability (returned) value
    // ----- temp variables
    long long int thisTask;                                   // task based approach for expose loop with small granularity
    // (good for shared memory parallelism)
    long long int numTasks=qureg.numAmpsPerChunk>>1;

    // ---------------------------------------------------------------- //
    //            dimensions                                            //
    // ---------------------------------------------------------------- //
    sizeHalfBlock = 1LL << (measureQubit);                       // number of state vector elements to sum,
    // and then the number to skip
    sizeBlock     = 2LL * sizeHalfBlock;                           // size of blocks (pairs of measure and skip entries)

    // ---------------------------------------------------------------- //
    //            find probability                                      //
    // ---------------------------------------------------------------- //

    //
    // --- task-based shared-memory parallel implementation
    //
    renorm=1/sqrt(totalProbability);

    thisTask = blockIdx.x*blockDim.x + threadIdx.x;
    if (thisTask>=numTasks) return;
    thisBlock = thisTask / sizeHalfBlock;
    index     = thisBlock*sizeBlock + thisTask%sizeHalfBlock;

    if (outcome==0){
        qureg.deviceStateVec[index].real=qureg.deviceStateVec[index].real*renorm;
        qureg.deviceStateVec[index].imag=qureg.deviceStateVec[index].imag*renorm;

        qureg.deviceStateVec[index+sizeHalfBlock].real=0;
        qureg.deviceStateVec[index+sizeHalfBlock].imag=0;
    } else if (outcome==1){
        qureg.deviceStateVec[index].real=0;
        qureg.deviceStateVec[index].imag=0;

        qureg.deviceStateVec[index+sizeHalfBlock].real=qureg.deviceStateVec[index+sizeHalfBlock].real*renorm;
        qureg.deviceStateVec[index+sizeHalfBlock].imag=qureg.deviceStateVec[index+sizeHalfBlock].imag*renorm;
    }
}

/*
 * outcomeProb must accurately be the probability of that qubit outcome in the state-vector, or
 * else the state-vector will lose normalisation
 */
void statevec_collapseToKnownProbOutcome(Qureg qureg, const int measureQubit, int outcome, qreal outcomeProb)
{
    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil((qreal)(qureg.numAmpsPerChunk>>1)/threadsPerCUDABlock);
    statevec_collapseToKnownProbOutcomeKernel<<<CUDABlocks, threadsPerCUDABlock>>>(qureg, measureQubit, outcome, outcomeProb);
}

/** Maps thread ID to a |..0..><..0..| state and then locates |0><1|, |1><0| and |1><1| */
__global__ void densmatr_collapseToKnownProbOutcomeKernel(
    qreal outcomeProb, Complex* stateVec, long long int numBasesToVisit,
    long long int part1, long long int part2, long long int part3,
    long long int rowBit, long long int colBit, long long int desired, long long int undesired)
{
    long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (scanInd >= numBasesToVisit) return;

    long long int base = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2);

    // renormalise desired outcome
    stateVec[base + desired].real /= outcomeProb;
    stateVec[base + desired].imag /= outcomeProb;

    // kill undesired outcome
    stateVec[base + undesired].real = 0;
    stateVec[base + undesired].imag = 0;

    // kill |..0..><..1..| states
    stateVec[base + colBit].real = 0;
    stateVec[base + colBit].imag = 0;
    stateVec[base + rowBit].real = 0;
    stateVec[base + rowBit].imag = 0;
}

/** This involves finding |...i...><...j...| states and killing those where i!=j */
void densmatr_collapseToKnownProbOutcome(Qureg qureg, const int measureQubit, int outcome, qreal outcomeProb) {

    int rowQubit = measureQubit + qureg.numQubitsRepresented;

    int colBit = 1LL << measureQubit;
    int rowBit = 1LL << rowQubit;

    long long int numBasesToVisit = qureg.numAmpsPerChunk/4;
	long long int part1 = colBit -1;
	long long int part2 = (rowBit >> 1) - colBit;
	long long int part3 = numBasesToVisit - (rowBit >> 1);

    long long int desired, undesired;
    if (outcome == 0) {
        desired = 0;
        undesired = colBit | rowBit;
    } else {
        desired = colBit | rowBit;
        undesired = 0;
    }

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil(numBasesToVisit / (qreal) threadsPerCUDABlock);
    densmatr_collapseToKnownProbOutcomeKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        outcomeProb, qureg.deviceStateVec, numBasesToVisit,
        part1, part2, part3, rowBit, colBit, desired, undesired);
}

__global__ void densmatr_addDensityMatrixKernel(Qureg combineQureg, qreal otherProb, Qureg otherQureg, long long int numAmpsToVisit) {

    long long int ampInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (ampInd >= numAmpsToVisit) return;

    combineQureg.deviceStateVec[ampInd].real *= 1-otherProb;
    combineQureg.deviceStateVec[ampInd].imag *= 1-otherProb;

    combineQureg.deviceStateVec[ampInd].real += otherProb*otherQureg.deviceStateVec[ampInd].real;
    combineQureg.deviceStateVec[ampInd].imag += otherProb*otherQureg.deviceStateVec[ampInd].imag;
}

void densmatr_addDensityMatrix(Qureg combineQureg, qreal otherProb, Qureg otherQureg) {

    long long int numAmpsToVisit = combineQureg.numAmpsPerChunk;

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
    densmatr_addDensityMatrixKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        combineQureg, otherProb, otherQureg, numAmpsToVisit
    );
}

/** Called once for every 4 amplitudes in density matrix
 * Works by establishing the |..0..><..0..| state (for its given index) then
 * visiting |..1..><..0..| and |..0..><..1..|. Labels |part1 X pa><rt2 NOT(X) part3|
 * From the brain of Simon Benjamin
 */
__global__ void densmatr_oneQubitDephaseKernel(
    qreal fac, Complex* stateVec, long long int numAmpsToVisit,
    long long int part1, long long int part2, long long int part3,
    long long int colBit, long long int rowBit)
{
    long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (scanInd >= numAmpsToVisit) return;

    long long int ampInd = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2);
    stateVec[ampInd + colBit].real *= fac;
    stateVec[ampInd + colBit].imag *= fac;
    stateVec[ampInd + rowBit].real *= fac;
    stateVec[ampInd + rowBit].imag *= fac;
}


void densmatr_oneQubitDegradeOffDiagonal(Qureg qureg, const int targetQubit, qreal dephFac) {

    long long int numAmpsToVisit = qureg.numAmpsPerChunk/4;

    int rowQubit = targetQubit + qureg.numQubitsRepresented;
    long long int colBit = 1LL << targetQubit;
    long long int rowBit = 1LL << rowQubit;

    long long int part1 = colBit - 1;
    long long int part2 = (rowBit >> 1) - colBit;
    long long int part3 = numAmpsToVisit - (rowBit >> 1);

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
    densmatr_oneQubitDephaseKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        dephFac, qureg.deviceStateVec, numAmpsToVisit,
        part1, part2, part3, colBit, rowBit);
}

void densmatr_oneQubitDephase(Qureg qureg, const int targetQubit, qreal dephase) {

    if (dephase == 0)
        return;

    qreal dephFac = 1 - dephase;
    densmatr_oneQubitDegradeOffDiagonal(qureg, targetQubit, dephFac);
}

/** Called 12 times for every 16 amplitudes in density matrix
 * Each sums from the |..0..0..><..0..0..| index to visit either
 * |..0..0..><..0..1..|,  |..0..0..><..1..0..|,  |..0..0..><..1..1..|,  |..0..1..><..0..0..|
 * etc and so on to |..1..1..><..1..0|. Labels |part1 0 part2 0 par><t3 0 part4 0 part5|.
 * From the brain of Simon Benjamin
 */
__global__ void densmatr_twoQubitDephaseKernel(
    qreal fac, Complex* stateVec, long long int numBackgroundStates, long long int numAmpsToVisit,
    long long int part1, long long int part2, long long int part3, long long int part4, long long int part5,
    long long int colBit1, long long int rowBit1, long long int colBit2, long long int rowBit2)
{
    long long int outerInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (outerInd >= numAmpsToVisit) return;

    // sets meta in 1...14 excluding 5, 10, creating bit string DCBA for |..D..C..><..B..A|
    int meta = 1 + (outerInd/numBackgroundStates);
    if (meta > 4) meta++;
    if (meta > 9) meta++;

    long long int shift = rowBit2*((meta>>3)%2) + rowBit1*((meta>>2)%2) + colBit2*((meta>>1)%2) + colBit1*(meta%2);
    long long int scanInd = outerInd % numBackgroundStates;
    long long int stateInd = (
        shift +
        (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2) + ((scanInd&part4)<<3) + ((scanInd&part5)<<4));

    stateVec[stateInd].real *= fac;
    stateVec[stateInd].imag *= fac;
}

// @TODO is separating these 12 amplitudes really faster than letting every 16th base modify 12 elems?
void densmatr_twoQubitDephase(Qureg qureg, int qubit1, int qubit2, qreal dephase) {

    if (dephase == 0)
        return;

    // assumes qubit2 > qubit1

    int rowQubit1 = qubit1 + qureg.numQubitsRepresented;
    int rowQubit2 = qubit2 + qureg.numQubitsRepresented;

    long long int colBit1 = 1LL << qubit1;
    long long int rowBit1 = 1LL << rowQubit1;
    long long int colBit2 = 1LL << qubit2;
    long long int rowBit2 = 1LL << rowQubit2;

    long long int part1 = colBit1 - 1;
    long long int part2 = (colBit2 >> 1) - colBit1;
    long long int part3 = (rowBit1 >> 2) - (colBit2 >> 1);
    long long int part4 = (rowBit2 >> 3) - (rowBit1 >> 2);
    long long int part5 = (qureg.numAmpsPerChunk/16) - (rowBit2 >> 3);
    qreal dephFac = 1 - dephase;

    // refers to states |a 0 b 0 c><d 0 e 0 f| (target qubits are fixed)
    long long int numBackgroundStates = qureg.numAmpsPerChunk/16;

    // 12 of these states experience dephasing
    long long int numAmpsToVisit = 12 * numBackgroundStates;

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
    densmatr_twoQubitDephaseKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        dephFac, qureg.deviceStateVec, numBackgroundStates, numAmpsToVisit,
        part1, part2, part3, part4, part5, colBit1, rowBit1, colBit2, rowBit2);
}

/** Works like oneQubitDephase but modifies every other element, and elements are averaged in pairs */
__global__ void densmatr_oneQubitDepolariseKernel(
    qreal depolLevel, Complex* stateVec, long long int numAmpsToVisit,
    long long int part1, long long int part2, long long int part3,
    long long int bothBits)
{
    long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (scanInd >= numAmpsToVisit) return;

    long long int baseInd = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2);
    long long int targetInd = baseInd + bothBits;

    qreal realAvDepol = depolLevel * 0.5 * (stateVec[baseInd].real + stateVec[targetInd].real);
    qreal imagAvDepol = depolLevel * 0.5 * (stateVec[baseInd].imag + stateVec[targetInd].imag);

    stateVec[baseInd].real   *= 1 - depolLevel;
    stateVec[baseInd].imag   *= 1 - depolLevel;
    stateVec[targetInd].real *= 1 - depolLevel;
    stateVec[targetInd].imag *= 1 - depolLevel;

    stateVec[baseInd].real   += realAvDepol;
    stateVec[baseInd].imag   += imagAvDepol;
    stateVec[targetInd].real += realAvDepol;
    stateVec[targetInd].imag += imagAvDepol;
}

/** Works like oneQubitDephase but modifies every other element, and elements are averaged in pairs */
__global__ void densmatr_oneQubitDampingKernel(
    qreal damping, Complex* stateVec, long long int numAmpsToVisit,
    long long int part1, long long int part2, long long int part3,
    long long int bothBits)
{
    long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (scanInd >= numAmpsToVisit) return;

    long long int baseInd = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2);
    long long int targetInd = baseInd + bothBits;

    qreal realAvDepol = damping  * ( stateVec[targetInd].real);
    qreal imagAvDepol = damping  * ( stateVec[targetInd].imag);

    stateVec[targetInd].real *= 1 - damping;
    stateVec[targetInd].imag *= 1 - damping;

    stateVec[baseInd].real   += realAvDepol;
    stateVec[baseInd].imag   += imagAvDepol;
}

void densmatr_oneQubitDepolarise(Qureg qureg, const int targetQubit, qreal depolLevel) {

    if (depolLevel == 0)
        return;

    densmatr_oneQubitDephase(qureg, targetQubit, depolLevel);

    long long int numAmpsToVisit = qureg.numAmpsPerChunk/4;
    int rowQubit = targetQubit + qureg.numQubitsRepresented;

    long long int colBit = 1LL << targetQubit;
    long long int rowBit = 1LL << rowQubit;
    long long int bothBits = colBit | rowBit;

    long long int part1 = colBit - 1;
    long long int part2 = (rowBit >> 1) - colBit;
    long long int part3 = numAmpsToVisit - (rowBit >> 1);

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
    densmatr_oneQubitDepolariseKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        depolLevel, qureg.deviceStateVec, numAmpsToVisit,
        part1, part2, part3, bothBits);
}

void densmatr_oneQubitDamping(Qureg qureg, const int targetQubit, qreal damping) {

    if (damping == 0)
        return;

    qreal dephase = sqrt(1-damping);
    densmatr_oneQubitDegradeOffDiagonal(qureg, targetQubit, dephase);

    long long int numAmpsToVisit = qureg.numAmpsPerChunk/4;
    int rowQubit = targetQubit + qureg.numQubitsRepresented;

    long long int colBit = 1LL << targetQubit;
    long long int rowBit = 1LL << rowQubit;
    long long int bothBits = colBit | rowBit;

    long long int part1 = colBit - 1;
    long long int part2 = (rowBit >> 1) - colBit;
    long long int part3 = numAmpsToVisit - (rowBit >> 1);

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
    densmatr_oneQubitDampingKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        damping, qureg.deviceStateVec, numAmpsToVisit,
        part1, part2, part3, bothBits);
}

/** Called once for every 16 amplitudes */
__global__ void densmatr_twoQubitDepolariseKernel(
    qreal depolLevel, Complex* stateVec, long long int numAmpsToVisit,
    long long int part1, long long int part2, long long int part3,
    long long int part4, long long int part5,
    long long int rowCol1, long long int rowCol2)
{
    long long int scanInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (scanInd >= numAmpsToVisit) return;

    // index of |..0..0..><..0..0|
    long long int ind00 = (scanInd&part1) + ((scanInd&part2)<<1) + ((scanInd&part3)<<2) + ((scanInd&part4)<<3) + ((scanInd&part5)<<4);
    long long int ind01 = ind00 + rowCol1;
    long long int ind10 = ind00 + rowCol2;
    long long int ind11 = ind00 + rowCol1 + rowCol2;

    qreal realAvDepol = depolLevel * 0.25 * (
        stateVec[ind00].real + stateVec[ind01].real + stateVec[ind10].real + stateVec[ind11].real);
    qreal imagAvDepol = depolLevel * 0.25 * (
        stateVec[ind00].imag + stateVec[ind01].imag + stateVec[ind10].imag + stateVec[ind11].imag);

    qreal retain = 1 - depolLevel;
    stateVec[ind00].real *= retain; stateVec[ind00].imag *= retain;
    stateVec[ind01].real *= retain; stateVec[ind01].imag *= retain;
    stateVec[ind10].real *= retain; stateVec[ind10].imag *= retain;
    stateVec[ind11].real *= retain; stateVec[ind11].imag *= retain;

    stateVec[ind00].real += realAvDepol; stateVec[ind00].imag += imagAvDepol;
    stateVec[ind01].real += realAvDepol; stateVec[ind01].imag += imagAvDepol;
    stateVec[ind10].real += realAvDepol; stateVec[ind10].imag += imagAvDepol;
    stateVec[ind11].real += realAvDepol; stateVec[ind11].imag += imagAvDepol;
}

void densmatr_twoQubitDepolarise(Qureg qureg, int qubit1, int qubit2, qreal depolLevel) {

    if (depolLevel == 0)
        return;

    // assumes qubit2 > qubit1

    densmatr_twoQubitDephase(qureg, qubit1, qubit2, depolLevel);

    int rowQubit1 = qubit1 + qureg.numQubitsRepresented;
    int rowQubit2 = qubit2 + qureg.numQubitsRepresented;

    long long int colBit1 = 1LL << qubit1;
    long long int rowBit1 = 1LL << rowQubit1;
    long long int colBit2 = 1LL << qubit2;
    long long int rowBit2 = 1LL << rowQubit2;

    long long int rowCol1 = colBit1 | rowBit1;
    long long int rowCol2 = colBit2 | rowBit2;

    long long int numAmpsToVisit = qureg.numAmpsPerChunk/16;
    long long int part1 = colBit1 - 1;
    long long int part2 = (colBit2 >> 1) - colBit1;
    long long int part3 = (rowBit1 >> 2) - (colBit2 >> 1);
    long long int part4 = (rowBit2 >> 3) - (rowBit1 >> 2);
    long long int part5 = numAmpsToVisit - (rowBit2 >> 3);

    int threadsPerCUDABlock, CUDABlocks;
    threadsPerCUDABlock = 128;
    CUDABlocks = ceil(numAmpsToVisit / (qreal) threadsPerCUDABlock);
    densmatr_twoQubitDepolariseKernel<<<CUDABlocks, threadsPerCUDABlock>>>(
        depolLevel, qureg.deviceStateVec, numAmpsToVisit,
        part1, part2, part3, part4, part5, rowCol1, rowCol2);
}

void seedQuESTDefault(){
    // init MT random number generator with three keys -- time and pid
    // for the MPI version, it is ok that all procs will get the same seed as random numbers will only be
    // used by the master process

    unsigned long int key[2];
    getQuESTDefaultSeedKey(key);
    init_by_array(key, 2);
}


#ifdef __cplusplus
}
#endif
