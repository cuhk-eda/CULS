#include <vector>
#include "robin_hood.h"
#include "common.h"
#include "aig/truth.cuh"

namespace Aig {

__host__ __device__ void getElemTruthTableInt(unsigned * vTruthElem, int nVars) {
    unsigned masks[5] = { 0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000 };
    int nWords = dUtils::TruthWordNum(nVars);
    unsigned * pTruth;

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

__host__ __device__ void getElemTruthTableInt64(uint64 * vTruthElem, int nVars) {
    // A: 1010, C: 1100, F: 1100
    uint64 masks[6] = { 0xAAAAAAAAAAAAAAAA, 0xCCCCCCCCCCCCCCCC, 0xF0F0F0F0F0F0F0F0, 0xFF00FF00FF00FF00, 0xFFFF0000FFFF0000, 0xFFFFFFFF00000000 };
    int nWords = dUtils::Truth6WordNum(nVars);
    uint64 * pTruth;

    for (int i = 0; i < nVars; i++) {
        pTruth = vTruthElem + i * nWords;
        if (i < 6) {
            for (int k = 0; k < nWords; k++)
                pTruth[k] = masks[i];
        } else {
            for (int k = 0; k < nWords; k++) {
                if (k & (1 << (i-6)))
                    pTruth[k] = ~(uint64)0;
                else
                    pTruth[k] = 0;
            }
        }
    }
}
/**
 * Get the elementary truth table (i.e., truth table of input variables).
 * Should be launched with only one thread. 
 **/
__global__ void getElemTruthTable(unsigned * vTruthElem, int nVars) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) // unsigned
        getElemTruthTableInt(vTruthElem, nVars);
}

__global__ void getElemTruthTable(uint64 * vTruthElem, int nVars) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0 ) // uint64
        getElemTruthTableInt64(vTruthElem, nVars);
}

__host__ void getElemTruthTableCPU(unsigned * vTruthElem, int nVars) {
    getElemTruthTableInt(vTruthElem, nVars);
}


__host__ void getCutTruthTableSingleCPU(const int * pFanin0, const int * pFanin1, 
                                        unsigned * pTruth, const unsigned * vTruthElem, 
                                        const int * pCut, int nVars, int rootId, 
                                        int nMaxCutSize, int nPIs) {
    int nodeId;
    robin_hood::unordered_set<int> sVisited;
    std::vector<int> vStack, vNodes;
    
    int nWords = dUtils::TruthWordNum(nVars);
    int nWordsElem = dUtils::TruthWordNum(nMaxCutSize);

    for (int i = 0; i < nVars; i++)
        sVisited.insert(pCut[i]);
    
    // 1. traversal to collect intermediate nodes
    vStack.push_back(rootId);
    while (!vStack.empty()) {
        // skip if already visited
        nodeId = vStack.back();
        vStack.pop_back();
        if (sVisited.count(nodeId) > 0)
            continue;
        
        assert(dUtils::AigIsNode(nodeId, nPIs));
        sVisited.insert(nodeId);

        // save result. make sure the nodes in vNodes are in reversed topo order (decreasing id)
        int j = vNodes.size() - 1;
        vNodes.push_back(0); // allocate a new entry
        for (; j >= 0 && vNodes[j] < nodeId; j--) // insertion sort
            vNodes[j + 1] = vNodes[j];
        vNodes[j + 1] = nodeId;

        // push fanins into stack
        vStack.push_back(dUtils::AigNodeID(pFanin1[nodeId]));
        vStack.push_back(dUtils::AigNodeID(pFanin0[nodeId]));
    }
    assert(vNodes[0] == rootId);

    // 2. compute truth table
    int nIntNodes = vNodes.size();
    unsigned * vTruthMem = new unsigned[(nVars + nIntNodes) * nWords];
    int * vTruthId2Node = new int[nVars + nIntNodes];

    // collect elementary truth tables for the cut nodes
    int nTruthId2NodeSize = 0;
    for (int i = 0; i < nVars; i++) {
        for (int j = 0; j < nWords; j++)
            vTruthMem[i * nWords + j] = vTruthElem[i * nWordsElem + j];
        vTruthId2Node[nTruthId2NodeSize++] = pCut[i];
    }
    for (int i = nIntNodes - 1; i >= 0; i--) {
        cutTruthIter(vNodes[i], pFanin0, pFanin1, vTruthMem, 
                     vTruthId2Node, &nTruthId2NodeSize, nWords);
    }
    assert(vTruthId2Node[nTruthId2NodeSize - 1] == rootId);

    // copy the truth table of rootId to pTruth
    for (int i = 0; i < nWords; i++)
        pTruth[i] = vTruthMem[(nTruthId2NodeSize - 1) * nWords + i];

    delete [] vTruthMem;
    delete [] vTruthId2Node;
}

} // namespace Aig
