#include "common.h"
#include "aig/truth.cuh"

/**
 * Get the elementary truth table (i.e., truth table of input variables).
 * Should be launched with only one thread. 
 **/
__global__ void Aig::getElemTruthTable(unsigned * vTruthElem, int nVars) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned masks[5] = { 0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000 };
    unsigned * pTruth;

    if (idx == 0) {
        int nWords = dUtils::TruthWordNum(nVars);

        for (int i = 0; i < nVars; i++) {
            pTruth = vTruthElem + i * nWords;
            if (i < 5) {
                for (int k = 0; k < nWords; k++)
                    pTruth[k] = masks[i];
            } else {
                for (int k = 0; k < nWords; k++) {
                    if (k & (1 << (i-5)))
                        pTruth[k] = ~(unsigned)0;
                    else
                        pTruth[k] = 0;
                }
            }
        }
    }
}


