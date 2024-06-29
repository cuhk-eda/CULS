#pragma once
#include <tuple>
#include "common.h"

#define BreakWhen(w, first, last, condition)    for(w=first; w<last; w++)   if(condition)   break

std::tuple<int, int *, int *, int *, int*> 
resubPerform(bool fUseZeros, bool fUseConstr, bool fUpdateLevel, 
                    int cutSize, int addNodes,
                    int nObjs, int nPIs, int nPOs, int nNodes, 
                    int * d_pFanin0, int * d_pFanin1, int * d_pOuts, 
                    int * d_pNumFanouts, const int * d_pLevel, int verbose,
                    const int * pFanin0, const int * pFanin1, const int * pOuts);

__device__ void buildDecGraph(int* dec, int rootLit, int obj0Lit);

// 1-resub
__device__ void buildDecGraph(int* dec, int rootLit, int obj0Lit, int obj1Lit, int orGate);

// 2-resub rootLit -> nodeLIt, obj2Lit; nodeLit-> obj0Lit, obj1Lit
__device__ void buildDecGraph(int* dec, int rootLit, int obj0Lit, int obj1Lit, int obj2Lit, int orGate);

__device__ void buildDecGraph22(int* dec, int rootLit, int obj0Lit, int obj1Lit, int obj2Lit, int orGate);

__device__ void buildDecGraph(int* dec, int rootLit, int obj0Lit, int obj1Lit, int obj2Lit, int obj3Lit, int orGate);




__global__ void resubDivsC(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize, 
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* vUsedNodes);

__global__ void resubDivs0(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize, 
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* vUsedNodes);

__global__ void resubDivs1(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize,
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* UPGlobal, int* UNGlobal, int* BIGlobal, int* queueSize, int* vUsedNodes);

__global__ void resubDivs12(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize,
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* UPGlobal, int* UNGlobal, int* BIGlobal, int* queueSize, const int* pLevel, int* vUsedNodes);

__global__ void resubDivs2(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize,
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* UPGlobal, int* UNGlobal, int* BIGlobal,
                            int* UP20Global, int* UP21Global, int* UN20Global, int* UN21Global,  
                            int* queueSize, int* vUsedNodes);
    
__global__ void resubDivs3(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize,
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose,
                            int* UP20Global, int* UP21Global, int* UN20Global, int* UN21Global,  
                            int* queueSize, int* vUsedNodes);


const int MAX_CUT_SIZE = 15;
const int CUT_TABLE_SIZE = 16;
const int STACK_SIZE = 512;
const int DIVISOR_SIZE = 150;
const int LARGE_STACK_SIZE = 2048;
// the size of MFFC is generally small
const int MAX_MFFC_SIZE = 64;
const int DIV2_MAX = 500;
const int DECGRAPH_SIZE = 8;
const int STATUS_VALID = 1;
const int STATUS_DELETE = 0;
const int STATUS_ROOT= -1;
const int STATUS_SKIP = -2;
const int INT_INF = 1000000000;
