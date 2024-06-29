
#pragma once
#include <tuple>
#include "common.h"
#include "resub.h"
#include "aig/truth.cuh"
#include "aig/strash.cuh"
#include "aig/traverse.cuh"

namespace Resub{

struct isNotSmallMffcResub {
    __host__ __device__
    bool operator()(const thrust::tuple<int, int> &e) const {
        return thrust::get<0>(e) == -1 || thrust::get<1>(e) >= 2;
    }
};


template <typename T>
struct ValidStatus {
    __host__ __device__
    bool operator()(const T &elem) {
        return elem == 1 || elem == -2 || elem == -1;
    }
};


struct descSortBy{
    const int* mask;
    descSortBy() = delete;
    descSortBy(const int* mask):
        mask(mask) {}

    __device__
    bool operator()(const int &lhs, const int& rhs){
        return mask[lhs] == mask[rhs] ? lhs > rhs : mask[lhs] > mask[rhs];
    }
};

template <int val>
struct unaryValue{
    __host__ __device__
    int operator()(const int &elem){
        return val;
    }
};



__global__ void printTruthTable(const int * pFanin0, const int * pFanin1,
                                 const int * vIndices, const int * vCutTable, const int * vCutSizes, 
                                 unsigned * vTruth, const int * vTruthRanges, const unsigned * vTruthElem,
                                 int nIndices, int nPIs, int nMaxCutSize){
    
    int rootId, nVars;

    for(int idx=0; idx<nIndices; idx++){
        rootId = vIndices[idx];
        nVars = vCutSizes[rootId];
        int nWords = dUtils::TruthWordNum(nMaxCutSize);
        printf("TT of node %d: ", rootId);
        Aig::printTruthTable(vTruth+rootId*nWords, nVars);
    }

}

__global__ void printIntTruth(int* vValidEnumInd, int nValid, int* vCutSizes, 
        int* intNodesInd, unsigned* intNodesTruth, int* intNodesSize, int nWords){
    
    for(int idx=0; idx<nValid; idx++){
        int rootId = vValidEnumInd[idx];
        int nVars = vCutSizes[rootId];
        int realWords = dUtils::TruthWordNum(nVars);
        printf("print the internal TT of node %3d:\n", rootId);
        for(int i=0; i<intNodesSize[idx]; i++){
            int nodeId = intNodesInd[idx*STACK_SIZE+i];
            printf("    TT of node %3d: ", nodeId);
            Aig::printTruthTable(intNodesTruth+idx*STACK_SIZE*nWords+i*realWords, nVars);
        }
    }
}


__global__ void printLevel(const int * vLevelNodes, int nLevelNodes, int currLevel, const int* vMaskValid){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx==0){
        printf("on the %d level: ", currLevel);
        for(int i=0; i <nLevelNodes; i++){
            int nodeId = vLevelNodes[i];
            printf(" %d(%d)", nodeId, vMaskValid[nodeId]);
        }
        printf("\n");
    }
}

__global__ void filterRedundNode(const int * vLevelNodes, int nLevelNodes, int * vMaskValid, int* vMFFCNumSaved, int nPIs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = gridDim.x * blockDim.x;
    for (; idx < nLevelNodes; idx += nThreads) {
        int nodeId = vLevelNodes[idx];
        if(dUtils::AigIsPIConst(nodeId, nPIs))
            vMaskValid[nodeId] = STATUS_SKIP;
        if(vMFFCNumSaved[nodeId]<1)
            vMaskValid[nodeId] = STATUS_SKIP;
        
    }

}

__global__ void printReconvCut(const int * pFanin0, const int * pFanin1, 
                             const int * pNumFanouts, const int * pLevels, 
                             const int * vReconvInd, int * vCutTable, int * vCutSizes, 
                             int nReconv, int nPIs, int nMaxCutSize, int nFanoutLimit) {
    for (int idx=0; idx < nReconv; idx++) {
        int nodeId = vReconvInd[idx];
        printf("cut of node %d(%d):", nodeId, vCutSizes[idx]);
        int cutSize = vCutSizes[idx];
        for(int j=0; j<cutSize; j++){
            printf(" %d", vCutTable[idx*CUT_TABLE_SIZE+j]);
        }
        printf("\n");
    }
}


}   // namespace Resub


__device__ int decrementRefResub(int idx, int rootId, int nodeId, int nPIs, int nMaxCutSize, int nMinLevel,
                            const int * pNumFanouts, const int * pLevels, 
                            int * vCutTable, int * vCutSizes, 
                            int * pTravSize, int * travIds, int * travRefs) {
    if (dUtils::AigIsPIConst(nodeId, nPIs) || pLevels[nodeId] <= nMinLevel) {
        // stop expansion, also do not add into the trav list
        int oldCutSize = vCutSizes[idx];
        if (oldCutSize < nMaxCutSize) {
            // add into cut list if it is not inside
            for (int i = 0; i < oldCutSize; i++)
                if (vCutTable[idx * CUT_TABLE_SIZE + i] == nodeId)
                    return 1;
            vCutTable[idx * CUT_TABLE_SIZE + oldCutSize] = nodeId;
            vCutSizes[idx]++;
            return 1;
        } else {
            // the cut has reached max size
            return -100;
        }
    }

    // check whether nodeId is already in the trav list
    for (int i = 0; i < *pTravSize; i++)
        if (travIds[i] == nodeId)
            return --travRefs[i];
    assert(*pTravSize < STACK_SIZE);

    // nodeId is not in the trav list; insert it
    travIds[*pTravSize] = nodeId;
    travRefs[*pTravSize] = pNumFanouts[nodeId] - 1;
    (*pTravSize)++;
    return pNumFanouts[nodeId] - 1;
}

// get the MFFC without boundary for the maximum memory to allocate
__global__ void getMffcNumResub(const int * pFanin0, const int * pFanin1, 
                           const int * pNumFanouts, const int * pLevels, 
                           int * vMFFCCutTable, int * vMFFCCutSizes, int * vConeSizes,
                           int nNodes, int nPIs, int nMaxCutSize) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stack[STACK_SIZE], travIds[STACK_SIZE], travRefs[STACK_SIZE];
    int stackTop, travSize, coneSize;
    int nodeId, rootId, faninId, nMinLevel;
    int fDecRet;

    for (; idx < nNodes; idx += nThreads) {
        stackTop = -1, travSize = 0, coneSize = 0;

        rootId = idx + nPIs + 1;
        vMFFCCutSizes[idx] = 0;

        stack[++stackTop] = rootId; // do not launch threads for PIs
        nMinLevel = max(0, pLevels[rootId] - 10);

        while (stackTop != -1) {
            nodeId = stack[stackTop--];
            // this is different bwtween global parallel MFFC evaluation and top-down level-wise evaluation
            coneSize++;

            // check its two fanins
            faninId = dUtils::AigNodeID(pFanin1[nodeId]);
            fDecRet = decrementRefResub(idx, rootId, faninId, nPIs, nMaxCutSize, nMinLevel, 
                                   pNumFanouts, pLevels, vMFFCCutTable, vMFFCCutSizes, 
                                   &travSize, travIds, travRefs);
            if (fDecRet == -100)
                break;             // cut size reached maximum
            else if (fDecRet == 0)
                stack[++stackTop] = faninId;
            assert(stackTop < STACK_SIZE);

            faninId = dUtils::AigNodeID(pFanin0[nodeId]);
            fDecRet = decrementRefResub(idx, rootId, faninId, nPIs, nMaxCutSize, nMinLevel, 
                                   pNumFanouts, pLevels, vMFFCCutTable, vMFFCCutSizes, 
                                   &travSize, travIds, travRefs);
            if (fDecRet == -100)
                break;
            else if (fDecRet == 0 )
                stack[++stackTop] = faninId;
            assert(stackTop < STACK_SIZE);

        }
        coneSize  = coneSize >= MAX_MFFC_SIZE ? MAX_MFFC_SIZE : coneSize;
        assert(coneSize<=MAX_MFFC_SIZE);
        vConeSizes[rootId] = coneSize;


    }
}


// get the MFFC insides the cut
__global__ void getMffcCutResub(const int * pFanin0, const int * pFanin1, const int * pNumFanouts, const int * pLevels, 
                           int* vValidEnumInd, int* vCutTable, int* vCutSizes, int nValid,
                           int * vMFFCCutTable, int * vMFFCCutSizes, int * vConeSizes,
                           int nNodes, int nPIs, int nMaxCutSize, int * vIntMFFCNodes, int* vIntMFFCNodesInd, int* vMFFCMark) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stack[STACK_SIZE], travIds[STACK_SIZE], travRefs[STACK_SIZE];
    int stackTop, travSize, coneSize;
    int nodeId, rootId, faninId, nMinLevel;
    int fDecRet;
    int* vIntMFFCNodesLocal;

    for (; idx < nValid; idx += nThreads) {
        stackTop = -1, travSize = 0, coneSize = 0;

        // rootId = idx + nPIs + 1;
        rootId = vValidEnumInd[idx];
        if(rootId<=nPIs){
            vMFFCCutSizes[idx] = vConeSizes[rootId] = 0;
            continue;
        }
        vMFFCCutSizes[idx] = 0;
        int* vCutTableLocal = vCutTable + idx*CUT_TABLE_SIZE;
        int cutSize = vCutSizes[idx];
        vIntMFFCNodesLocal = vIntMFFCNodes + vIntMFFCNodesInd[rootId-1];

        stack[++stackTop] = rootId; // do not launch threads for PIs
        nMinLevel = max(0, pLevels[rootId] - 10);



        while (stackTop != -1) {
            nodeId = stack[stackTop--];
            vIntMFFCNodesLocal[coneSize] = nodeId;
            // this is different bwtween global parallel MFFC evaluation and top-down level-wise evaluation
            atomicCAS(vMFFCMark+nodeId, nodeId, rootId);
            if(rootId!=nodeId)    atomicMin(vMFFCMark+nodeId, rootId);
            coneSize++;

            // check its two fanins
            faninId = dUtils::AigNodeID(pFanin1[nodeId]);
            fDecRet = decrementRefResub(idx, rootId, faninId, nPIs, nMaxCutSize, nMinLevel, 
                                   pNumFanouts, pLevels, vMFFCCutTable, vMFFCCutSizes, 
                                   &travSize, travIds, travRefs);
            if (fDecRet == -100)
                break;             // cut size reached maximum
            else if (fDecRet == 0 && !dUtils::hasVisited(vCutTableLocal, faninId, cutSize))
                stack[++stackTop] = faninId;
            assert(stackTop < STACK_SIZE);

            faninId = dUtils::AigNodeID(pFanin0[nodeId]);
            fDecRet = decrementRefResub(idx, rootId, faninId, nPIs, nMaxCutSize, nMinLevel, 
                                   pNumFanouts, pLevels, vMFFCCutTable, vMFFCCutSizes, 
                                   &travSize, travIds, travRefs);
            if (fDecRet == -100)
                break;
            else if (fDecRet == 0 && !dUtils::hasVisited(vCutTableLocal, faninId, cutSize))
                stack[++stackTop] = faninId;
            assert(stackTop < STACK_SIZE);

        }
        assert(coneSize<MAX_MFFC_SIZE);
        vConeSizes[rootId] = coneSize;

    }
}




__device__ int getReconvCutIterResub(int rootIdx,
                                const int * pFanin0, const int * pFanin1, 
                                const int * pNumFanouts, const int * pLevels, 
                                int * visited, int * pVisitedSize,
                                int * vCutTable, int * vCutSizes, 
                                int nPIs, int nMaxCutSize, int nFanoutLimit) {
    int nodeId, faninId, bestId = -1, bestIdx = -1;
    int bestCost = 100, currCost;
    int fFanin0Visited, fFanin1Visited, fBestFanin0Visited = 0, fBestFanin1Visited = 0;
    int rootId = rootIdx;   // the rootId is in fact the idx of the root node in one level

    // find the best cost cut node to expand
    for (int i = 0; i < vCutSizes[rootId]; i++) {
        nodeId = vCutTable[rootId * CUT_TABLE_SIZE + i];

        // get the number of new leaves
        fFanin0Visited = fFanin1Visited = 0;
        if (dUtils::AigIsPIConst(nodeId, nPIs))
            currCost = 999;
        else {
            faninId = dUtils::AigNodeID(pFanin0[nodeId]);
            for (int j = 0; j < *pVisitedSize; j++)
                if (visited[j] == faninId) {
                    fFanin0Visited = 1;
                    break;
                }
            
            faninId = dUtils::AigNodeID(pFanin1[nodeId]);
            for (int j = 0; j < *pVisitedSize; j++)
                if (visited[j] == faninId) {
                    fFanin1Visited = 1;
                    break;
                }
            
            currCost = (1 - fFanin0Visited) + (1 - fFanin1Visited);
            if (currCost >= 2) {
                if (pNumFanouts[nodeId] > nFanoutLimit)
                    currCost = 999;
            }
        }

        // update best node
        if (bestCost > currCost || (bestCost == currCost && pLevels[nodeId] > pLevels[bestId])) {
            bestCost = currCost, bestId = nodeId, bestIdx = i;
            fBestFanin0Visited = fFanin0Visited, fBestFanin1Visited = fFanin1Visited;
        }
        if (bestCost == 0)
            break;
    }

    if (bestId == -1)
        return 0;
    assert(bestCost < 3);

    if (vCutSizes[rootId] - 1 + bestCost > nMaxCutSize)
        return 0;
    assert(dUtils::AigIsNode(bestId, nPIs));
    // remove the best node from cut list
    for (int i = bestIdx + 1; i < vCutSizes[rootId]; i++)
        vCutTable[rootId * CUT_TABLE_SIZE + i - 1] = vCutTable[rootId * CUT_TABLE_SIZE + i];
    vCutSizes[rootId]--;

    if (!fBestFanin0Visited) {
        assert(*pVisitedSize < STACK_SIZE);
        faninId = dUtils::AigNodeID(pFanin0[bestId]);
        vCutTable[rootId * CUT_TABLE_SIZE + (vCutSizes[rootId]++)] = faninId;
        visited[(*pVisitedSize)++] = faninId;
    }
    if (!fBestFanin1Visited) {
        assert(*pVisitedSize < STACK_SIZE);
        faninId = dUtils::AigNodeID(pFanin1[bestId]);
        vCutTable[rootId * CUT_TABLE_SIZE + (vCutSizes[rootId]++)] = faninId;
        visited[(*pVisitedSize)++] = faninId;
    }
    assert(vCutSizes[rootId] <= nMaxCutSize);
    return 1;
}

__global__ void getReconvCutResub(const int * pFanin0, const int * pFanin1, 
                             const int * pNumFanouts, const int * pLevels, 
                             const int * vReconvInd, int * vCutTable, int * vCutSizes, 
                             int nReconv, int nPIs, int nMaxCutSize, int nFanoutLimit,
                             int* intNodesInd, int* intNodesSize) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int visited[STACK_SIZE];
    int visitedSize;
    int rootId, faninId;
    int* vCutTableLocal;

    for (; idx < nReconv; idx += nThreads) {
        visitedSize = 0;
        rootId = vReconvInd[idx];
        if(rootId<=nPIs){
            vCutSizes[idx] = 0;
            continue;
        }

        vCutSizes[idx] = 0;
        vCutTableLocal = vCutTable + idx*CUT_TABLE_SIZE;

        // initialize the cut list and visited list
        visited[visitedSize++] = rootId;

        faninId = dUtils::AigNodeID(pFanin0[rootId]);
        vCutTableLocal[(vCutSizes[idx]++)] = faninId;
        visited[visitedSize++] = faninId;

        faninId = dUtils::AigNodeID(pFanin1[rootId]);
        vCutTableLocal[(vCutSizes[idx]++)] = faninId;
        visited[visitedSize++] = faninId;

        // iteratively expand the cut
        while (getReconvCutIterResub(idx, pFanin0, pFanin1, pNumFanouts, pLevels, visited, &visitedSize,
                   vCutTable, vCutSizes, nPIs, nMaxCutSize, nFanoutLimit));
        intNodesSize[idx] = visitedSize;
        memcpy(intNodesInd+idx*STACK_SIZE, visited, sizeof(int)*visitedSize);
        // intNodesInd has to be arranged in reverse topo(descending) order
        assert(vCutSizes[idx] <= nMaxCutSize);
    }
}




inline int isRedundantNode(int nodeId, int nPIs, const int * fanin0) {
    return nodeId > nPIs && fanin0[nodeId] == dUtils::AigConst1;
}

int topoSortGetLevelResub(int nodeId, int nPIs, int * levels, const int * fanin0, const int * fanin1) {
    assert(nodeId <= nPIs || fanin0[nodeId] != -1);

    if (levels[nodeId] != -1)
        return levels[nodeId];
    if (isRedundantNode(nodeId, nPIs, fanin0)){
        return (levels[nodeId] = 
                topoSortGetLevelResub(AigNodeID(fanin1[nodeId]), nPIs, levels, fanin0, fanin1));
    }
    return (levels[nodeId] = 
            1 + max(
                topoSortGetLevelResub(AigNodeID(fanin0[nodeId]), nPIs, levels, fanin0, fanin1),
                topoSortGetLevelResub(AigNodeID(fanin1[nodeId]), nPIs, levels, fanin0, fanin1)
            ));
}

std::tuple<int, int *, int *, int *, int>
reorderResub(int* vFanin0New, int* vFanin1New, int* pOutsNew, int nPIs, int nPOs, int nBufferLen){
    int * vhFanin0, * vhFanin1, * vhLevels, * vhNewInd;
    int * vhFanin0New, * vhFanin1New, * vhOutsNew;
    int * pOuts;

    int nNodesNew, nObjsNew;
    // copy fanin arrays to host
    cudaHostAlloc(&vhFanin0, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhFanin1, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhLevels, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&vhNewInd, nBufferLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&pOuts, sizeof(int) * nPOs, cudaHostAllocDefault);

    cudaMemcpy(vhFanin0, vFanin0New, nBufferLen * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vhFanin1, vFanin1New, nBufferLen * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pOuts, pOutsNew, sizeof(int)*nPOs, cudaMemcpyDeviceToHost);
    memset(vhLevels, -1, nBufferLen * sizeof(int));
    cudaDeviceSynchronize();
    for (int i = 0; i <= nPIs; i++)
        vhLevels[i] = 0;
    for (int i = nPIs + 1; i < nBufferLen; i++)
        if (vhFanin0[i] != -1) {
            topoSortGetLevelResub(i, nPIs, vhLevels, vhFanin0, vhFanin1);
        }
    // count total number of nodes and assign each node an id level by level
    int nMaxLevel = 0;
    std::vector<int> vLevelNodesCount(1, 0);
    for (int i = nPIs + 1; i < nBufferLen; i++)
        if (vhFanin0[i] != -1 && !isRedundantNode(i, nPIs, vhFanin0)) {
            assert(vhLevels[i] > 0);
            if (vhLevels[i] > nMaxLevel) {
                while (vLevelNodesCount.size() < vhLevels[i] + 1)
                    vLevelNodesCount.push_back(0);
                nMaxLevel = vhLevels[i];
            }
            assert(vhLevels[i] < vLevelNodesCount.size());
            vLevelNodesCount[vhLevels[i]]++;
        }
    assert(vLevelNodesCount[0] == 0);
printf("maxLevel: %d\n", nMaxLevel);
    
    for (int i = 1; i <= nMaxLevel; i++)
        vLevelNodesCount[i] += vLevelNodesCount[i - 1];
    nNodesNew = vLevelNodesCount.back();
    nObjsNew = nNodesNew + nPIs + 1;
    
    // assign consecutive new ids
    for (int i = nBufferLen - 1; i > nPIs; i--)
        if (vhFanin0[i] != -1 && !isRedundantNode(i, nPIs, vhFanin0))
            vhNewInd[i] = (--vLevelNodesCount[vhLevels[i]]) + nPIs + 1;
    // ids for PIs do not change
    for (int i = 0; i <= nPIs; i++)
        vhNewInd[i] = i;

    // gather nodes in assigned order
    vhFanin0New = (int *) malloc(nObjsNew * sizeof(int));
    vhFanin1New = (int *) malloc(nObjsNew * sizeof(int));
    vhOutsNew = (int *) malloc(nPOs * sizeof(int));
    memset(vhFanin0New, -1, nObjsNew * sizeof(int));
    memset(vhFanin1New, -1, nObjsNew * sizeof(int));

    for (int i = nPIs + 1; i < nBufferLen; i++)
        if (vhFanin0[i] != -1 && !isRedundantNode(i, nPIs, vhFanin0)) {
            assert(vhFanin0New[vhNewInd[i]] == -1 && vhFanin1New[vhNewInd[i]] == -1);
            // propagate if fanin is redundant
            int lit, propLit = vhFanin0[i];
            while(isRedundantNode(AigNodeID(propLit), nPIs, vhFanin0))
                propLit = dUtils::AigNodeNotCond(vhFanin1[AigNodeID(propLit)], AigNodeIsComplement(propLit));
            lit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(propLit)], AigNodeIsComplement(propLit));

            vhFanin0New[vhNewInd[i]] = lit;

            propLit = vhFanin1[i];
            while(isRedundantNode(AigNodeID(propLit), nPIs, vhFanin0))
                propLit = dUtils::AigNodeNotCond(vhFanin1[AigNodeID(propLit)], AigNodeIsComplement(propLit));
            lit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(propLit)], AigNodeIsComplement(propLit));
            vhFanin1New[vhNewInd[i]] = lit;

            if (vhFanin0New[vhNewInd[i]] > vhFanin1New[vhNewInd[i]]) {
                int temp = vhFanin0New[vhNewInd[i]];
                vhFanin0New[vhNewInd[i]] = vhFanin1New[vhNewInd[i]];
                vhFanin1New[vhNewInd[i]] = temp;
            }
        }
  
    // update POs
    for (int i = 0; i < nPOs; i++) {
        int oldId = AigNodeID(pOuts[i]);
        int lit, propLit;
        assert(oldId <= nPIs || vhFanin0[oldId] != -1);

        propLit = pOuts[i];
        while(isRedundantNode(AigNodeID(propLit), nPIs, vhFanin0))
            propLit = dUtils::AigNodeNotCond(vhFanin1[AigNodeID(propLit)], AigNodeIsComplement(propLit));
        lit = dUtils::AigNodeLitCond(vhNewInd[AigNodeID(propLit)], AigNodeIsComplement(propLit));
        
        vhOutsNew[i] = lit;
    }

    cudaFreeHost(vhFanin0);
    cudaFreeHost(vhFanin1);
    cudaFreeHost(vhLevels);
    cudaFreeHost(vhNewInd);

    return {nObjsNew, vhFanin0New, vhFanin1New, vhOutsNew, nMaxLevel};

}




__device__ int calcRemainMFFCSingle(int rootId, int tarNodeId, int* vMaskValid, int* vMFFCNodes, int MFFCSize, int* d_pFanin0, int* d_pFanin1, int* rmMFFC){
    int remain = 0;
    assert(rootId == vMFFCNodes[0]);
    if( vMaskValid[rootId]==STATUS_DELETE) return 0;
    int localDelete[STACK_SIZE], localMFFC[STACK_SIZE];   // store the remaining part in the MFFC
    memset(localDelete, 0, sizeof(int)*MFFCSize);
    memcpy(localMFFC, vMFFCNodes, sizeof(int)*MFFCSize);
    for(int i=0; i<MFFCSize; i++){
        int nodeId = vMFFCNodes[i];
        assert(vMaskValid[nodeId]!=STATUS_ROOT);
        assert(vMaskValid[nodeId]!=STATUS_DELETE);  
        if(nodeId == tarNodeId || localDelete[i]){
            int left = dUtils::AigNodeID(d_pFanin0[nodeId]);
            int right = dUtils::AigNodeID(d_pFanin1[nodeId]);
            for(int j=i+1; j<MFFCSize; j++){
                localDelete[j] = (localMFFC[j]==left || localMFFC[j]==right) ? 1 : localDelete[j];
            }
            rmMFFC[remain++] = nodeId;
        }

    }
    return remain;
}

// print the detailed subgraph and conflict relation
__global__ void printDetailResub(int nObjs, int nPIs, int* vMaskValid, int* vIntMFFCNodes, int* vIntMFFCNodesInd, int* vMFFCNumSaved, int* decGraph, int* vGain, int* d_pFanin0, int* d_pFanin1){
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rmMFFC[STACK_SIZE];
    for (; idx < nObjs; idx += nThreads) {
        if(dUtils::AigIsPIConst(idx, nPIs)) continue;
        if(!vGain[idx]) continue;
        int rootId = idx;
        int MFFCsize = vMFFCNumSaved[rootId];
        int divNum = 0;
        // int* vIntMFFCLocal = vIntMFFCNodes + rootId * MAX_MFFC_SIZE;
        int* vIntMFFCLocal = vIntMFFCNodes + vIntMFFCNodesInd[rootId-1];
        int* decGraphLocal = decGraph + rootId * DECGRAPH_SIZE;
        assert(MFFCsize - decGraphLocal[0] == vGain[rootId]);
        printf("%d\n", rootId);
        // printf("%d gain: %d - %d = %d\n", rootId, MFFCsize, decGraphLocal[0], vGain[rootId]);
        // print decGraph
        printf("%d ", decGraphLocal[0]);
        for(int i=decGraphLocal[0]+1; i<=decGraphLocal[0]*2+1; i++)
            printf("%d ", dUtils::AigNodeID(decGraphLocal[i]));
        printf("\n");

        // print MFFC
        printf("%d ", MFFCsize);
        for(int i=0; i<MFFCsize; i++){
            printf("%d ", vIntMFFCLocal[i]);
            if(vMaskValid[vIntMFFCLocal[i]]==STATUS_SKIP)   divNum++;
        }
        printf("\n");
        // print MFFC of inner div
        printf("%d\n", divNum);
        for(int i=0; i<MFFCsize; i++){
            int nodeId = vIntMFFCLocal[i];
            if(vMaskValid[nodeId]!=STATUS_SKIP) continue;
            int remain = calcRemainMFFCSingle(rootId, nodeId, vMaskValid, vIntMFFCLocal, MFFCsize, d_pFanin0, d_pFanin1, rmMFFC);
            assert(remain);
            assert(nodeId == rmMFFC[0]);
            printf("%d ", remain);
            for(int j=0; j<remain; j++)
                printf("%d ", rmMFFC[j]);
            printf("\n");
        }
        printf("\n");
    }   
}

__global__ void printStatistics(int nPIs, int nObjs, int nNodes, int addNodes, int* vGain, int* decGraph, int* vMaskValid, int totalCandNum, int nCandRoot, int* vUsedNodes){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx==0){
        int totalGain = 0, gains[4] = {0}, constGain = 0;
        int totalRoot = 0, roots[4] = {0}, constRoot = 0;
        int usedNodesTotal[10] = {0}, usedNodes[10] = {0};
        for(int i=nPIs+1; i<nObjs; i++){
            int rootId = i;
            int* decGraphLocal = decGraph+rootId*DECGRAPH_SIZE;
            if(vUsedNodes[rootId]!=-1)    usedNodesTotal[ vUsedNodes[rootId] ]++;
            if(vMaskValid[rootId]!=STATUS_ROOT) continue;
            usedNodes[ vUsedNodes[rootId] ]++;
            int resubType = decGraphLocal[0];
            totalGain += vGain[rootId];
            if(resubType==0 && decGraphLocal[1]<=1){
                constGain += vGain[rootId];
                constRoot++;
            }
            else    
                gains[resubType] += vGain[rootId];
            totalRoot ++;
            roots[resubType] ++;
        }

        printf("************* Statictics *************\n");
        printf("  original inputs: %d, ands: %d\n", nPIs, nObjs);
        printf("  valid roots: %d, gains: %d\n", totalRoot, totalGain);
        printf("    const:   roots: %d, gains: %d\n", constRoot, constGain);
        for(int i=0; i<=addNodes; i++){
            printf("    %d-resub: ", i);
            printf("roots: %d, gains: %d\n", roots[i], gains[i]);
        }
        printf("  nodes: %d - %d = %d\n", nNodes, totalGain, nNodes - totalGain);
        printf("  all roots: %d\n", totalCandNum);
        printf("    no conflict: %d, conflict: %d\n", totalCandNum-nCandRoot, nCandRoot);
        printf("  resub type:\n");
        printf("    const      %3d/%3d\n", usedNodes[0], usedNodesTotal[0]);
        printf("    replace    %3d/%3d\n", usedNodes[1], usedNodesTotal[1]);
        printf("    single or  %3d/%3d\n", usedNodes[2], usedNodesTotal[2]);
        printf("    single and %3d/%3d\n", usedNodes[3], usedNodesTotal[3]);
        printf("    double or  %3d/%3d\n", usedNodes[4], usedNodesTotal[4]);
        printf("    double and %3d/%3d\n", usedNodes[5], usedNodesTotal[5]);
        printf("    Or-And     %3d/%3d\n", usedNodes[6], usedNodesTotal[6]);
        printf("    And-Or     %3d/%3d\n", usedNodes[7], usedNodesTotal[7]);
        printf("    Or-2And    %3d/%3d\n", usedNodes[8], usedNodesTotal[8]);
        printf("    And-2Or    %3d/%3d\n", usedNodes[9], usedNodesTotal[9]);
        printf("**************** Done ****************\n");


    }
}