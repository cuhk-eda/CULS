#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <tuple>
#include <functional>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include "common.h"
#include "aig_manager.h"
#include "hash_table.h"
#include "resub.h"
#include "resub_utils.h"
// #include "refactor.cu"
#include "aig/truth.cuh"
#include "aig/strash.cuh"
#include "aig/traverse.cuh"
#include "algorithms/sop/sop.cuh"
#include "misc/print.cuh"
#include <unistd.h>


using namespace Resub;

extern __global__ void printDetailResub(int nObjs, int nPIs, int* vMaskValid, int* vIntMFFCNodes, int* vMFFCNumSaved, int* decGraph, int* vGain, int* d_pFanin0, int* d_pFanin1);
extern std::tuple<int, int *, int *, int *, int> reorderResub(int* vFanin0New, int* vFanin1New, int* pOutsNew, int nPIs, int nPOs, int nBufferLen);
extern __global__ void getReconvCutResub(const int * pFanin0, const int * pFanin1, 
                             const int * pNumFanouts, const int * pLevels, 
                             const int * vReconvInd, int * vCutTable, int * vCutSizes, 
                             int nReconv, int nPIs, int nMaxCutSize, int nFanoutLimit,
                             int* intNodesInd, int* intNodesSize);
extern __global__ void getMffcCutResub(const int * pFanin0, const int * pFanin1, 
                           const int * pNumFanouts, const int * pLevels, 
                           int* vValidEnumInd, int* vCutTable, int* vCutSizes, int nValid,
                           int * vMFFCCutTable, int * vMFFCCutSizes, int * vConeSizes,
                           int nNodes, int nPIs, int nMaxCutSize, int * vIntMFFCNodes, int* vMFFCMark);
extern __global__ void getMffcNumResub(const int * pFanin0, const int * pFanin1, 
                           const int * pNumFanouts, const int * pLevels, 
                           int * vMFFCCutTable, int * vMFFCCutSizes, int * vConeSizes,
                           int nNodes, int nPIs, int nMaxCutSize);


__global__ void collectDivisor(const int * pFanin0, const int * pFanin1, const int* d_pNumFanouts, const int* d_pLevel, int nPIs, int nObjs,
                            const int* vValidEnumInd, const int nValid, int* intNodesInd, const int* intNodesSize, int* vCutTable, int* vCutSizes,
                            int* vIntMFFCNodes, int* vIntMFFCNodesInd, const int* vMFFCNumSaved, int* divisor, int* divisorSizes, int* simSizes,
                            int* vFanoutRanges, int* vFanoutInd, int nFanoutLimit, int* vMaskValid, int* vReverseLevel, int* vMFFCMark, 
                            bool fUseConstr, bool fUpdateLevel, int levelMax){

    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId, j, k;
    int* intNodesLocal, * divisorLocal, * vIntMFFCNodesLocal, * vCutTableLocal;
    int divSize, simSize;

    for (; idx < nValid; idx += nThreads) {
        divSize = simSize = 0;
        rootId = vValidEnumInd[idx];
        int MFFCSize = vMFFCNumSaved[rootId];
        if(MFFCSize==0){
            divisorSizes[idx] = 0;
            simSizes[idx] = 0;
            continue;
        }
        assert(rootId);
        divisorLocal = divisor + idx * DIVISOR_SIZE;
        intNodesLocal = intNodesInd + idx*STACK_SIZE;
        vCutTableLocal = vCutTable + idx * CUT_TABLE_SIZE;
        // if(MFFCSize > vIntMFFCNodesInd[rootId] - vIntMFFCNodesInd[rootId-1])
        //     printf("assert72failed %d %d %d %d\n", rootId, MFFCSize, vIntMFFCNodesInd[rootId], vIntMFFCNodesInd[rootId-1]);
        assert(MFFCSize <= vIntMFFCNodesInd[rootId] - vIntMFFCNodesInd[rootId-1]);
        vIntMFFCNodesLocal = vIntMFFCNodes + vIntMFFCNodesInd[rootId-1];
        int coneSize = intNodesSize[idx];

        // printf("rootId %d\n", rootId);
        int rootReverseLevel = vReverseLevel[rootId];
        int required = fUpdateLevel ? levelMax - rootReverseLevel : INT_INF;
        // assert(coneSize>=MFFCSize);
        // add the cuts into divisors
        int cutSize = vCutSizes[idx];
        for(int i=0; i<cutSize; i++)
            divisorLocal[i] = vCutTableLocal[i];
        divSize = cutSize;
        // add the internal nodes into divisors
        for(int i=coneSize-1; i>=0; i--){
            int candNodeId = intNodesLocal[i];
            // skip if the node is in Cut or MFFC
            if(dUtils::hasVisited(vCutTableLocal, candNodeId, cutSize) || dUtils::hasVisited(vIntMFFCNodesLocal, candNodeId, MFFCSize))    
                continue;
            // insertion sort (increasing order)
            for(k = divSize-1; k>=cutSize && divisorLocal[k] > candNodeId; k--)
                divisorLocal[k+1] = divisorLocal[k];
            divisorLocal[k+1] = candNodeId;
            divSize++;
        }
        assert(divSize+MFFCSize<DIVISOR_SIZE);
        // expand divisor
        // TODO: here is the point to speed up
        for(int i=0; i<divSize; i++){
            int divId = divisorLocal[i];
            if(d_pNumFanouts[divId]>nFanoutLimit)   continue;
            for(int j=(divId==0)?0:vFanoutRanges[divId-1]; j<vFanoutRanges[divId]; j++){
                int fanoutId = vFanoutInd[j];
                if(fanoutId>=nObjs)  continue;
                if(vMaskValid[fanoutId]==STATUS_DELETE || d_pLevel[fanoutId] >= required)  continue;
                // if(vMaskValid[fanoutId]==STATUS_DELETE || d_pLevel[fanoutId] >= d_pLevel[rootId])  continue;
                if(fUseConstr){  
                    // method 1
                    if(d_pLevel[fanoutId]>d_pLevel[rootId]  )   continue;
                    if(d_pLevel[fanoutId]==d_pLevel[rootId] && fanoutId > rootId)   continue;
                    // method 2
                    // if(d_pLevel[fanoutId]>=d_pLevel[rootId])   continue;
                }
                int otherFaninId = (dUtils::AigNodeID(pFanin0[fanoutId])==divId) ? 
                                    dUtils::AigNodeID(pFanin1[fanoutId]) : dUtils::AigNodeID(pFanin0[fanoutId]);
                // if(otherFaninId==rootId || !dUtils::hasVisited(divisorLocal, otherFaninId, divSize)) continue;
                if(otherFaninId==rootId) continue;
                if(dUtils::hasVisited(vIntMFFCNodesLocal, fanoutId, MFFCSize)) continue;
                bool ifSuc=false;
                for(int k=0; k<divSize; k++){
                    if(divisorLocal[k]==fanoutId){  ifSuc = false;  break;  }
                    if(divisorLocal[k]==otherFaninId) ifSuc=true;
                }
                if(!ifSuc) continue;
                // if(dUtils::hasVisited(vIntMFFCNodesLocal, fanoutId, MFFCSize) || dUtils::hasVisited(divisorLocal, fanoutId, divSize) ) continue;
                divisorLocal[divSize++] = fanoutId;
                if(divSize+MFFCSize>=DIVISOR_SIZE-1)  break;
            }
            if(divSize+MFFCSize>=DIVISOR_SIZE-1)  break;
        }

        assert(divSize+MFFCSize<DIVISOR_SIZE);
        divisorSizes[idx] = divSize;
        simSize = divSize;

        // add the MFFC node into divisors for simulation, but divisorSize does not count them
        // attention that MFFC might exceed the cut
        for(int i=MFFCSize-1; i>=0; i--){
            int mffcNode = vIntMFFCNodesLocal[i];
            for(j=simSize-1; j>=divSize && divisorLocal[j] > mffcNode; j--)
                divisorLocal[j+1] = divisorLocal[j];
            divisorLocal[j+1] = mffcNode;
            simSize++;
            assert(simSize<STACK_SIZE);
        }
        simSizes[idx] = simSize;

        // assert(simSize == divSize + MFFCSize);
        // if(divisorLocal[simSize-1]!=rootId)
        //     printf("assert failed138 %d %d %d %d\n", rootId, MFFCSize, divSize, simSize);
        // assert(divisorLocal[simSize-1]==rootId);

        // if(rootId==241){
        //     printf("\nprint the cut of rootId %d(%d):", rootId, d_pLevel[rootId]);
        //     for(int i=0; i<cutSize; i++)
        //         printf(" %d", vCutTableLocal[i]);
        //     printf("\nprint the intNodes:");
        //     for(int i=0; i<intNodesSize[idx]; i++)
        //         printf(" %d", intNodesLocal[i]);
        //     printf("\nprint the MFFC:");
        //     for(int i=0; i<vMFFCNumSaved[rootId]; i++)
        //         printf(" %d", vIntMFFCNodesLocal[i]);
        //     printf("\nprint the divisor:");
        //     for(int i=0; i<divisorSizes[idx]; i++)
        //         printf(" %d(%d)", divisorLocal[i], d_pLevel[divisorLocal[i]]);
        //     printf(" |");
        //     for(int i=divisorSizes[idx]; i<simSizes[idx]; i++)
        //         printf(" %d(%d)", divisorLocal[i], d_pLevel[divisorLocal[i]]);
        //     printf("\n");
        // }
        //  if(divisorLocal[simSize-1]!=rootId)
        //     printf("assert failed138 %d %d %d %d\n", rootId, MFFCSize, divSize, simSize);
        assert(divisorLocal[simSize-1]==rootId);

       
    }                            
    
}

__global__ void simulateResub(const int * pFanin0, const int * pFanin1,
                                 const int * vIndices, const int * vCutTable, const int * vCutSizes, 
                                 unsigned * vTruth, const unsigned * vTruthElem,
                                 int nIndices, int nPIs, int nMaxCutSize,
                                 int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize,
                                 int* nodePhase) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId, nVars;
    int nWords, nWordsElem = dUtils::TruthWordNum(nMaxCutSize);

    int visited[STACK_SIZE], visitedSize;

    unsigned* divTruthLocal;
    int* divisorLocal, *nodePhaseLocal;


    for (; idx < nIndices; idx += nThreads) {
        rootId = vIndices[idx];
        nVars = vCutSizes[idx];
        nWords = dUtils::TruthWordNum(nVars);
        divisorLocal = divisor + idx*DIVISOR_SIZE;
        divTruthLocal = divisorTruth + idx*nWordsElem*DIVISOR_SIZE; 
        nodePhaseLocal = nodePhase + idx*DIVISOR_SIZE;
        if(simSize[idx]==0) continue;
        assert(simSize[idx]<DIVISOR_SIZE);

        visitedSize = 0;
        for(int i=0; i<nVars; i++){
            for (int j = 0; j < nWords; j++){
                divTruthLocal[i * nWords + j] = vTruthElem[i * nWordsElem + j];
                // assert(i*nWordsElem+j>=0);
                // assert(i*nWordsElem+j < nMaxCutSize * nWordsElem);
            }
            visited[visitedSize++] = divisorLocal[i];
        }
        for(int i=nVars; i<simSize[idx]; i++){
            // printf("rootId %d, simSize %d, i %d, %d\n", rootId, simSize[idx], i, divisorLocal[i]);
            Aig::cutTruthIter(divisorLocal[i], pFanin0, pFanin1, divTruthLocal, visited, &visitedSize, nWords);
        }
        assert(visited[visitedSize - 1] == rootId);

        // normalize, the highest bit in TT is alway 0
        for(int i=0; i<simSize[idx]; i++){
            if(divTruthLocal[i*nWords] & 1){
                nodePhaseLocal[i] = 1;
                // printf("normalizing rootId %d, %d %d\n", rootId, divisorLocal[i], nodePhaseLocal[i]);
                for (int j = 0; j < nWords; j++)
                    divTruthLocal[i * nWords + j] = ~divTruthLocal[i * nWords + j];
            }
            else
                nodePhaseLocal[i] = 0;
        }

        
        for (int i = 0; i < nWords; i++)
            *(vTruth+idx*nWordsElem+i) = divTruthLocal[(visitedSize - 1) * nWords + i];


    }
}

__host__ __device__ 
void markMFFCdeleted(int rootId, int* vMaskValid, int* pMFFCLocal, const int MFFCsize, int addNodes){
    // the array must be global w.r.t rootId, not level index
    for(int i=0; i<MFFCsize; i++){
        assert( vMaskValid[pMFFCLocal[i]] == STATUS_VALID || vMaskValid[pMFFCLocal[i]] == STATUS_SKIP );
        vMaskValid[pMFFCLocal[i]] = STATUS_DELETE;
    }
    vMaskValid[rootId] = STATUS_ROOT;
}


__device__ int traceBackResub(int rootLIt, int* decGraph, int *vMaskValid){
    int newLit = rootLIt;
    int newId = dUtils::AigNodeID(rootLIt);
    // if rootLit is a root 0-resub, trace downwards
    while(decGraph[newId*DECGRAPH_SIZE]==0 && vMaskValid[newId]==STATUS_ROOT){
        int nextLit = decGraph[newId*DECGRAPH_SIZE+1];
        // newLit = dUtils::AigNodeNotCond(nextLit, dUtils::AigNodeIsComplement(newLit)^dUtils::AigNodeIsComplement(nextLit));
        newLit = dUtils::AigNodeNotCond(nextLit, dUtils::AigNodeIsComplement(newLit));
        newId = dUtils::AigNodeID(newLit);
        assert(newId!=dUtils::AigNodeID(rootLIt));  // might be circle: a->b b->c c->a
    }
    newLit = dUtils::AigNodeNotCond(newLit, dUtils::AigNodeIsComplement(decGraph[newId*DECGRAPH_SIZE+1]) && vMaskValid[newId]==STATUS_ROOT);
    return newLit;
}


__global__ void reconstruct(int* d_pFanin0, int* d_pFanin1, int* d_pOuts, int nObjs, int* vValidEnumInd, int nValid, int nPOs,
                            int* decGraph, int* vIntMFFCNodes, int* vIntMFFCNodesInd, int* vMFFCNumSaved, int* vMaskValid, const int* d_pLevel){
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId, MFFCSize;
    int* pMFFCLocal, *decGraphLocal;
    for (; idx < nValid; idx += nThreads) {
        rootId = vValidEnumInd[idx];
        if(idx<nPOs){
            d_pOuts[idx] = traceBackResub(d_pOuts[idx], decGraph, vMaskValid);
        }
        if(vMaskValid[rootId]==STATUS_DELETE)   continue;
        if(vMaskValid[rootId]!=STATUS_ROOT && vMaskValid[rootId]!=STATUS_DELETE){
            int newFaninLit0 = traceBackResub(d_pFanin0[rootId], decGraph, vMaskValid);
            int newFaninLit1 = traceBackResub(d_pFanin1[rootId], decGraph, vMaskValid);
            d_pFanin0[rootId] = newFaninLit0;
            d_pFanin1[rootId] = newFaninLit1;
            continue;
        }
        MFFCSize = vMFFCNumSaved[rootId];
        // pMFFCLocal = vIntMFFCNodes + rootId*MAX_MFFC_SIZE;
        pMFFCLocal = vIntMFFCNodes + vIntMFFCNodesInd[rootId-1];
        decGraphLocal = decGraph + rootId * DECGRAPH_SIZE;
        assert(MFFCSize > decGraphLocal[0]);
        // change internal structure
        int resubType = decGraphLocal[0];
        if(resubType==0){
            // do not need to reconstruct
        }
        if(resubType==1){
            assert(rootId == dUtils::AigNodeID(decGraphLocal[1]));
            assert(rootId == pMFFCLocal[0]);
            if(vMaskValid[dUtils::AigNodeID(decGraphLocal[2])]==STATUS_DELETE ||vMaskValid[dUtils::AigNodeID(decGraphLocal[3])]==STATUS_DELETE )
                printf("assert1204failed %d %d %d\n", rootId, dUtils::AigNodeID(decGraphLocal[2]), dUtils::AigNodeID(decGraphLocal[3]));
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[2])]!=STATUS_DELETE);
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[3])]!=STATUS_DELETE);
            int newFaninLit0 = traceBackResub(decGraphLocal[2], decGraph, vMaskValid);
            int newFaninLit1 = traceBackResub(decGraphLocal[3], decGraph, vMaskValid);
            d_pFanin0[rootId] = newFaninLit0;
            d_pFanin1[rootId] = newFaninLit1;
        }
        if(resubType==2){
            // mark the mask back to 1
            assert(rootId == dUtils::AigNodeID(decGraphLocal[1]));
            assert(rootId == pMFFCLocal[0]);
            if(vMaskValid[dUtils::AigNodeID(decGraphLocal[2])]!=STATUS_DELETE)
                printf("assert1217failed rootId %d, nodeId %d\n", rootId, dUtils::AigNodeID(decGraphLocal[2]));
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[2])]==STATUS_DELETE);
            // vMaskValid[dUtils::AigNodeID(decGraphLocal[2])] = STATUS_VALID;
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[3])]!=STATUS_DELETE);
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[4])]!=STATUS_DELETE);
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[5])]!=STATUS_DELETE);
            assert(!( vMaskValid[d_pFanin0[rootId]>>1]==STATUS_SKIP && vMaskValid[d_pFanin1[rootId]>>1]==STATUS_SKIP  ));
            int newFaninLit0 = traceBackResub(decGraphLocal[2], decGraph, vMaskValid);
            int newFaninLit1 = traceBackResub(decGraphLocal[3], decGraph, vMaskValid);
            int newFaninLit2 = traceBackResub(decGraphLocal[4], decGraph, vMaskValid);
            int newFaninLit3 = traceBackResub(decGraphLocal[5], decGraph, vMaskValid);
            d_pFanin0[rootId] = newFaninLit0;
            d_pFanin1[rootId] = newFaninLit3;
            d_pFanin0[dUtils::AigNodeID(newFaninLit0)] = newFaninLit1;
            d_pFanin1[dUtils::AigNodeID(newFaninLit0)] = newFaninLit2;
// int id1 = dUtils::AigNodeID(newFaninLit1), id2 = dUtils::AigNodeID(newFaninLit2), id3 = dUtils::AigNodeID(newFaninLit3);
// int level1 = d_pLevel[id1], level2 = d_pLevel[id2], level3 = d_pLevel[id3]; 
// int rootLevel = d_pLevel[rootId];
// if(rootLevel-level1<=1 || rootLevel-level2 <=1)
//     printf("%d(%d) <- %d(%d) %d(%d) %d(%d)\n", rootId, rootLevel, id1, level1, id2, level2, id3, level3);

        }
        if(resubType==3){
            assert(rootId == dUtils::AigNodeID(decGraphLocal[1]));
            assert(rootId == pMFFCLocal[0]);
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[2])]==STATUS_DELETE);
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[3])]==STATUS_DELETE);
            // vMaskValid[dUtils::AigNodeID(decGraphLocal[2])] = STATUS_VALID;
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[4])]!=STATUS_DELETE);
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[5])]!=STATUS_DELETE);
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[6])]!=STATUS_DELETE);
            assert(vMaskValid[dUtils::AigNodeID(decGraphLocal[7])]!=STATUS_DELETE);
            assert(!( vMaskValid[d_pFanin0[rootId]>>1]==STATUS_SKIP && vMaskValid[d_pFanin1[rootId]>>1]==STATUS_SKIP  ));
            int newFaninLit1 = traceBackResub(decGraphLocal[2], decGraph, vMaskValid);
            int newFaninLit2 = traceBackResub(decGraphLocal[3], decGraph, vMaskValid);
            int newFaninLit3 = traceBackResub(decGraphLocal[4], decGraph, vMaskValid);
            int newFaninLit4 = traceBackResub(decGraphLocal[5], decGraph, vMaskValid);
            int newFaninLit5 = traceBackResub(decGraphLocal[6], decGraph, vMaskValid);
            int newFaninLit6 = traceBackResub(decGraphLocal[7], decGraph, vMaskValid);
            d_pFanin0[rootId] = newFaninLit1;
            d_pFanin1[rootId] = newFaninLit2;
            d_pFanin0[dUtils::AigNodeID(newFaninLit1)] = newFaninLit3;
            d_pFanin1[dUtils::AigNodeID(newFaninLit1)] = newFaninLit4;
            d_pFanin0[dUtils::AigNodeID(newFaninLit2)] = newFaninLit5;
            d_pFanin1[dUtils::AigNodeID(newFaninLit2)] = newFaninLit6;

        }

    }
}


__global__ void deleteZeroResub(int* vValidEnumInd, int nValid, int* decGraph, int* vMaskValid){
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId;
    int *decGraphLocal;
    for (; idx < nValid; idx += nThreads) {
        rootId = vValidEnumInd[idx];
        assert(rootId);
        assert(vMaskValid[rootId]==STATUS_ROOT);
        decGraphLocal = decGraph + rootId * DECGRAPH_SIZE;
        int resubType = decGraphLocal[0];
        if(resubType==0)
            vMaskValid[rootId] = STATUS_DELETE;
        else if(resubType==2)
            vMaskValid[dUtils::AigNodeID(decGraphLocal[2])] = STATUS_VALID;
        else if(resubType==3){
            vMaskValid[dUtils::AigNodeID(decGraphLocal[2])] = STATUS_VALID;
            vMaskValid[dUtils::AigNodeID(decGraphLocal[3])] = STATUS_VALID;
        }
    }
}


__global__ void remapNode(int* d_pFanin0, int* d_pFanin1, int nPIs, int nObjs, int nObjsNew, int* oldToNewNodeId, int* vFanin0New, int* vFanin1New, int* vNumFanouts, int* vMaskValid ){
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < nObjs; idx += nThreads) {
        int nodeId = idx;
// printf("%d(%d) -> %d\n", nodeId, vMaskValid[nodeId], oldToNewNodeId[nodeId]);
        if(dUtils::AigIsPIConst(nodeId, nPIs)){
            assert(nodeId==oldToNewNodeId[nodeId]-1);
            vFanin0New[nodeId] = -1;
            vFanin1New[nodeId] = -1;
            continue;
        }
        // if(vMaskValid[nodeId]!=STATUS_VALID && vMaskValid[nodeId]!=STATUS_SKIP)    continue;
        if(vMaskValid[nodeId]==STATUS_DELETE)   continue;
        int newNodeId = oldToNewNodeId[nodeId]-1;
// if(newNodeId == 76 || newNodeId == 75 || newNodeId == 79)
//     printf("remapNode %d -> %d\n", nodeId, newNodeId);
        assert(newNodeId<nObjsNew);
        int fanin0Lit = d_pFanin0[nodeId], fanin1Lit = d_pFanin1[nodeId];
        int fanin0Id = dUtils::AigNodeID(fanin0Lit), fanin1Id = dUtils::AigNodeID(fanin1Lit);
        int newFanin0Id = oldToNewNodeId[fanin0Id]-1, newFanin1Id = oldToNewNodeId[fanin1Id]-1;
        vFanin0New[newNodeId] = dUtils::AigNodeLitCond(newFanin0Id, dUtils::AigNodeIsComplement(fanin0Lit));
        vFanin1New[newNodeId] = dUtils::AigNodeLitCond(newFanin1Id, dUtils::AigNodeIsComplement(fanin1Lit));
        atomicAdd(&vNumFanouts[newFanin0Id], 1);
        atomicAdd(&vNumFanouts[newFanin1Id], 1);
        if(newNodeId==newFanin0Id || newNodeId==newFanin1Id)
            printf("assert failed ori: %d(%d) %d(%d) %d(%d), new: %d %d %d\n", nodeId, vMaskValid[nodeId] ,fanin0Id, vMaskValid[fanin0Id], fanin1Id,vMaskValid[fanin1Id], newNodeId, newFanin0Id, newFanin1Id);
        assert(newNodeId!=newFanin0Id);
        assert(newNodeId!=newFanin1Id);
    // printf("remap %d(mask %d) (%d %d) -> %d (%d %d)\n", nodeId, vMaskValid[nodeId], fanin0Id, fanin1Id, 
    //                                 newNodeId, newFanin0Id, newFanin1Id);
    }
}

__global__ void remapPO(int* d_pOuts, int* pOutsNew, int* vNumFanouts, int nPOs, int* oldToNewNodeId, int* vMaskValid){
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < nPOs; idx += nThreads) {
        int faninLit = d_pOuts[idx];
        int faninId = dUtils::AigNodeID(faninLit);
        int newFaninId = oldToNewNodeId[faninId]-1;
        // d_pOuts[idx] = dUtils::AigNodeLitCond(newFaninId, dUtils::AigNodeIsComplement(faninLit));
        pOutsNew[idx] = dUtils::AigNodeLitCond(newFaninId, dUtils::AigNodeIsComplement(faninLit));
        atomicAdd(&vNumFanouts[newFaninId], 1);
    }
}



// the rootId itself is not considered
__device__ __forceinline__ 
bool findPosGainMFFCPre(int rootId, int* vMFFCMark, int* vGain){
    if(rootId == vMFFCMark[rootId] || !vMFFCMark[rootId]) return false;
    int preNode = rootId;
    while(preNode = vMFFCMark[preNode]){
        if(vGain[preNode])  return true;
        if(preNode == vMFFCMark[preNode])   return false;
    }
    return false;
}

__global__ void checkMffcConflict(int* vCandRootId, int nCandRoot, int* vMaskValid, 
                            int* vIntMFFCNodes, int* vIntMFFCNodesInd, int* vMFFCNumSaved, int* vMFFCMark, 
                            int* decGraph, int* vGain, int verbose){
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < nCandRoot; idx += nThreads) {
        int rootId = vCandRootId[idx];
        assert(rootId);
        // int* vIntMFFCLocal = vIntMFFCNodes + rootId * MAX_MFFC_SIZE;
        int* vIntMFFCLocal = vIntMFFCNodes + vIntMFFCNodesInd[rootId-1] ;
        assert(rootId==vIntMFFCLocal[0]);
        int* decGraphLocal = decGraph + rootId * DECGRAPH_SIZE;
        int i, MffcSize = vMFFCNumSaved[rootId];
        // case1: if one of its MFFC nodes is another node's div, there is conflict
        for(i=0; i<MffcSize; i++){
            if(vMaskValid[ vIntMFFCLocal[i] ] == STATUS_SKIP || (i && vGain[vIntMFFCLocal[i]]))   break;
        }
        if(i!=MffcSize){
            continue;
            // printf("rootId %d, conflict type: 1\n", rootId);
        }
        // case2: if its div node is in another nodes' MFFC
        // decGraphLocal[0] = addNodes, decGraphLocal[1] = rootId
        for(i=decGraphLocal[0]+1; i<=decGraphLocal[0]*2+1; i++){
            int divNode = dUtils::AigNodeID(decGraphLocal[i]);
            // TODO: it is not applicable in fUseZero
            if(vGain[divNode] || findPosGainMFFCPre(divNode, vMFFCMark, vGain))   break;
        }
        if(i!=decGraphLocal[0]*2+2) {
            continue;
        }
        // case3: rootId is an MFFC node
        if(findPosGainMFFCPre(rootId, vMFFCMark, vGain)) {
            // printf("rootId %d, conflict type: 3\n", rootId);
            continue;
        }
        int gain = vMFFCNumSaved[rootId] - decGraphLocal[0];
        atomicAdd(vGain+0, gain);
        if(verbose>=3)  printf("no conflict %d, gain %d\n", rootId, gain);
        markMFFCdeleted(rootId, vMaskValid, vIntMFFCLocal, vMFFCNumSaved[rootId], decGraphLocal[0]);
        if(decGraphLocal[0]==2){
            decGraphLocal[2] = dUtils::AigNodeLitCond(*(vIntMFFCLocal+1), dUtils::AigNodeIsComplement(decGraphLocal[2]));
        }
        else if(decGraphLocal[0]==3){
            decGraphLocal[2] = dUtils::AigNodeLitCond(*(vIntMFFCLocal+1), dUtils::AigNodeIsComplement(decGraphLocal[2]));
            decGraphLocal[3] = dUtils::AigNodeLitCond(*(vIntMFFCLocal+2), dUtils::AigNodeIsComplement(decGraphLocal[3]));

        }
    }
}

// attention that MFFC must be in pre-order traverse
__host__ __device__ 
int calcRemainMFFCGreedy(int rootId, int* vMaskValid, int* vMFFCNodes, int MFFCSize, const int* d_pFanin0, const int* d_pFanin1, int* rmMFFC, int* solidDiv){
    int remain = 0;
    assert(rootId == vMFFCNodes[0]);
    // if(solidDiv[rootId] || vMaskValid[rootId]==STATUS_DELETE) return 0;
    if( vMaskValid[rootId]==STATUS_DELETE) return 0;
    assert(vMaskValid[rootId]!=STATUS_ROOT);
    int localDelete[STACK_SIZE], localMFFC[STACK_SIZE];   // store the remaining part in the MFFC
    memset(localDelete, 0, sizeof(int)*MFFCSize);
    memcpy(localMFFC, vMFFCNodes, sizeof(int)*MFFCSize);
    rmMFFC[remain++] = rootId;
    for(int i=1; i<MFFCSize; i++){
        int nodeId = vMFFCNodes[i];
        // if(vMaskValid[nodeId]==STATUS_ROOT || vMaskValid[nodeId]==STATUS_DELETE) printf("nodeId %d %d %d\n", rootId, nodeId, vMaskValid[nodeId]);
        // assert(vMaskValid[nodeId]!=STATUS_ROOT);
        // assert(vMaskValid[nodeId]!=STATUS_DELETE);   // this case might happen cause a node's MFFC might be a subset of its ancestor's
        if( solidDiv[nodeId] || vMaskValid[nodeId]==STATUS_ROOT || localDelete[i]){
            int left = dUtils::AigNodeID(d_pFanin0[nodeId]);
            int right = dUtils::AigNodeID(d_pFanin1[nodeId]);
            for(int j=i+1; j<MFFCSize; j++){
                localDelete[j] = (localMFFC[j]==left || localMFFC[j]==right) ? 1 : localDelete[j];
            }
        }
        else
            rmMFFC[remain++] = nodeId;
    }
    return remain;
}

   
bool loopDetector(int rootId, int travId, int nPIs, int* travNodes, const int* d_pFanin0, const int* d_pFanin1, int* vMaskValid, int* decGraph, int* stack){
    // int stack[LARGE_STACK_SIZE]; 
    int stackTop = -1, nodeId;
    int *decGraphLocal = decGraph + rootId*DECGRAPH_SIZE;
    for(int i=decGraphLocal[0]+1; i<decGraphLocal[0]*2+2; i++){
        stack[++stackTop] = dUtils::AigNodeID(decGraphLocal[i]);
        travNodes[dUtils::AigNodeID(decGraphLocal[i])] = travId;
    }

    while(stackTop!=-1){
        nodeId = stack[stackTop--];
        if(nodeId == rootId)    return true;
        if(nodeId<=nPIs)    continue;
        assert(travNodes[nodeId]==travId);
        decGraphLocal = decGraph + nodeId*DECGRAPH_SIZE;
        if(vMaskValid[nodeId]==STATUS_ROOT){
            for(int i=decGraphLocal[0]+1; i<decGraphLocal[0]*2+2; i++){
                int faninId = dUtils::AigNodeID(decGraphLocal[i]);
                if(travNodes[faninId]!=travId){
                    stack[++stackTop] = faninId;
                    travNodes[faninId] = travId;
                }
            }
        }
        else{
            int faninId =dUtils::AigNodeID(d_pFanin0[nodeId]);
            if(travNodes[faninId]!=travId){
                stack[++stackTop] = faninId;
                travNodes[faninId] = travId;
            }
            faninId =dUtils::AigNodeID(d_pFanin1[nodeId]);
            if(travNodes[faninId]!=travId){
                stack[++stackTop] = faninId;
                travNodes[faninId] = travId;
            }
        }

        assert(stackTop<LARGE_STACK_SIZE);

    }
    return false;

}


__global__ void resolveMffcConflictGreedyGPU(int* vCandRootId,int nCandRoot, int nObjs, int nPIs, int*  vMaskValid,int*  vIntMFFCNodes, int* vIntMFFCNodesInd, int nIntMFFCNodes,
                                            int* vMFFCNumSaved, int* decGraph, int* vGain, const int* d_pFanin0, const int* d_pFanin1, int* solidDiv, int verbose, bool fUseConstr){

int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx==0){
clock_t beginTime = clock();
clock_t time1, time2, time3;
printf("begin resolve conflict by GPU\n");

    int rmMFFC[STACK_SIZE];

time1 = clock();

    for (int idx=0; idx < nCandRoot; idx ++) {
        int rootId = vCandRootId[idx];
        assert(rootId>nPIs);
        int* vIntMFFCLocal = vIntMFFCNodes + vIntMFFCNodesInd[rootId-1];
        assert(rootId==vIntMFFCLocal[0]);
        int* decGraphLocal = decGraph + rootId * DECGRAPH_SIZE;
        int i, divNode;
        // 1. first check if the div nodes are valid
        if(vMaskValid[rootId]==STATUS_DELETE){
            if(verbose>=3)  printf("in resolving conflict rootId %d, has been deleted\n", rootId);
            continue;
        }
        // 2. 
        for(i=decGraphLocal[0]+1; i<=decGraphLocal[0]*2+1; i++){
            divNode = dUtils::AigNodeID(decGraphLocal[i]);
            // if(vMaskValid[divNode]==STATUS_DELETE || vMaskValid[divNode]==STATUS_ROOT)  break;
            if(vMaskValid[divNode]==STATUS_DELETE )  break;
        }
        if(i!=decGraphLocal[0]*2+2){
            if(verbose>=3)    printf("in resolving conflict rootId %d, skip\n", rootId);
            continue;
        }
        // 3. loop detect
        assert(fUseConstr);

        int remain = calcRemainMFFCGreedy(rootId, vMaskValid, vIntMFFCLocal, vMFFCNumSaved[rootId], d_pFanin0, d_pFanin1, rmMFFC, solidDiv);
        // if(verbose>=3)  printf("in resolving conflict rootId %d, ori MFFC size %d, remain %d\n", rootId, vMFFCNumSaved[rootId], remain);
        if(remain > decGraphLocal[0]){
            int gain = remain - decGraphLocal[0];
            if(verbose>=3)    printf("commit %d, gain %d = %d - %d\n", rootId, gain, remain, decGraphLocal[0]);
            vGain[rootId] = gain;
            markMFFCdeleted(rootId, vMaskValid, rmMFFC, remain, decGraphLocal[0]);

            for(i=decGraphLocal[0]+1; i<=decGraphLocal[0]*2+1; i++){
                divNode = dUtils::AigNodeID(decGraphLocal[i]);
                solidDiv[divNode] = 1;
            }   
            if(decGraphLocal[0]==2){
                decGraphLocal[2] = dUtils::AigNodeLitCond(rmMFFC[1], dUtils::AigNodeIsComplement(decGraphLocal[2]));
            }
            else if(decGraphLocal[0]==3){
                decGraphLocal[2] = dUtils::AigNodeLitCond(rmMFFC[1], dUtils::AigNodeIsComplement(decGraphLocal[2]));
                decGraphLocal[3] = dUtils::AigNodeLitCond(rmMFFC[2], dUtils::AigNodeIsComplement(decGraphLocal[3]));

            }
       }
        else if(verbose>=3) printf("in resolving conflict rootId %d, remain %d <= %d\n", rootId, remain, decGraphLocal[0]);

    }
time2= clock();

    // free(rmMFFC);

time3 = clock();
double total_time = (double)(time3 - beginTime);
printf("    takes %.2f %% sec to prepare\n", (time1 - beginTime) / total_time);
printf("    takes %.2f %% sec to run\n", (time2 - time1) / total_time);
printf("    takes %.2f %% sec to free\n", (time3 - time2) / total_time);

// printf("resolve conflict done, takes %.2f sec\n", (time3 - beginTime) / (double) CLOCKS_PER_SEC);
printf("resolve conflict done\n");

}   // if(idx==0)

}


void resolveMffcConflictGreedy(int* dCandRootId,int nCandRoot, int nObjs, int nPIs, int*  dMaskValid,int*  dIntMFFCNodes, int* dIntMFFCNodesInd, int nIntMFFCNodes,
                                            int* dMFFCNumSaved, int* ddecGraph, int* dGain, const int* d_pFanin0, const int* d_pFanin1, int verbose, bool fUseConstr){
clock_t beginTime = clock();
clock_t time1, time2, time3;
clock_t time11, time22;
clock_t phase1=0, phase2=0, phase3 = 0;
printf("begin resolve conflict\n");

    int *vCandRootId, *vMaskValid, *vIntMFFCNodes, *vIntMFFCNodesInd, *vMFFCNumSaved, *decGraph, *vGain;
    vCandRootId = (int*)malloc(sizeof(int)*nCandRoot); 
    vMaskValid = (int*)malloc(sizeof(int)*nObjs);
    vIntMFFCNodes = (int*)malloc(sizeof(int)*nIntMFFCNodes);
    vIntMFFCNodesInd = (int*)malloc(sizeof(int)*nObjs);
    vMFFCNumSaved = (int*)malloc(sizeof(int)*nObjs);
    vGain = (int*)malloc(sizeof(int)*nObjs);

    cudaMemcpy(vCandRootId, dCandRootId, sizeof(int)*nCandRoot, cudaMemcpyDeviceToHost);
    cudaMemcpy(vMaskValid, dMaskValid, sizeof(int)*nObjs, cudaMemcpyDeviceToHost);
    cudaMemcpy(vIntMFFCNodes, dIntMFFCNodes, sizeof(int)*nIntMFFCNodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vIntMFFCNodesInd, dIntMFFCNodesInd, sizeof(int)*nObjs, cudaMemcpyDeviceToHost);
    cudaMemcpy(vMFFCNumSaved, dMFFCNumSaved, sizeof(int)*nObjs, cudaMemcpyDeviceToHost);
    cudaMemcpy(vGain, dGain, sizeof(int)*nObjs, cudaMemcpyDeviceToHost);

time11 = clock();
phase1 = time11 - beginTime;
    cudaMallocHost(&decGraph, sizeof(int)*nObjs*DECGRAPH_SIZE);
time22 = clock();
phase2 = time22 - time11;
    cudaMemcpy(decGraph, ddecGraph, sizeof(int)*nObjs*DECGRAPH_SIZE, cudaMemcpyDeviceToHost);
    // decGraph = (int*)malloc(sizeof(int)*nObjs*DECGRAPH_SIZE);
    // cudaMemcpy(decGraph, ddecGraph, sizeof(int)*nObjs*DECGRAPH_SIZE, cudaMemcpyDeviceToHost);
time1 = clock();
phase3 = time1 - time22;
 

    int *solidDiv,  *travNodes; 
    int rmMFFC[STACK_SIZE];
    solidDiv = (int*)calloc(nObjs, sizeof(int));
    travNodes = (int*)calloc(nObjs, sizeof(int));

printf("    takes %.2f sec to prepare (%.3f + %.3f + %.3f)\n", 
    (time1 - beginTime) / (double) CLOCKS_PER_SEC, 
    phase1 / (double) CLOCKS_PER_SEC, phase2 / (double) CLOCKS_PER_SEC, phase3 / (double) CLOCKS_PER_SEC);

    for (int idx=0; idx < nCandRoot; idx ++) {
// time1 = clock();
        int rootId = vCandRootId[idx];
        assert(rootId>nPIs);
        // int* vIntMFFCLocal = vIntMFFCNodes + rootId * MAX_MFFC_SIZE;
        int* vIntMFFCLocal = vIntMFFCNodes + vIntMFFCNodesInd[rootId-1];
        assert(rootId==vIntMFFCLocal[0]);
        int* decGraphLocal = decGraph + rootId * DECGRAPH_SIZE;
        int i, divNode;
        // 1. first check if the div nodes are valid
        if(vMaskValid[rootId]==STATUS_DELETE){
            if(verbose>=2)  printf("in resolving conflict rootId %d, has been deleted\n", rootId);
            continue;
        }
        // 2. check whether its div still exsists
        for(i=decGraphLocal[0]+1; i<=decGraphLocal[0]*2+1; i++){
            divNode = dUtils::AigNodeID(decGraphLocal[i]);
            // if(vMaskValid[divNode]==STATUS_DELETE || vMaskValid[divNode]==STATUS_ROOT)  break;
            if(vMaskValid[divNode]==STATUS_DELETE )  break;
        }
        if(i!=decGraphLocal[0]*2+2){
            if(verbose>=2)    printf("in resolving conflict rootId %d, skip\n", rootId);
            continue;
        }

        int remain = calcRemainMFFCGreedy(rootId, vMaskValid, vIntMFFCLocal, vMFFCNumSaved[rootId], d_pFanin0, d_pFanin1, rmMFFC, solidDiv);

        if(remain > decGraphLocal[0]){
            int gain = remain - decGraphLocal[0];
            if(verbose>=2)    printf("commit %d, gain %d = %d - %d\n", rootId, gain, remain, decGraphLocal[0]);
            vGain[rootId] = gain;
            markMFFCdeleted(rootId, vMaskValid, rmMFFC, remain, decGraphLocal[0]); 
            for(i=decGraphLocal[0]+1; i<=decGraphLocal[0]*2+1; i++){
                divNode = dUtils::AigNodeID(decGraphLocal[i]);
                solidDiv[divNode] = 1;
            }   
            if(decGraphLocal[0]==2){
                decGraphLocal[2] = dUtils::AigNodeLitCond(rmMFFC[1], dUtils::AigNodeIsComplement(decGraphLocal[2]));
            }
            else if(decGraphLocal[0]==3){
                decGraphLocal[2] = dUtils::AigNodeLitCond(rmMFFC[1], dUtils::AigNodeIsComplement(decGraphLocal[2]));
                decGraphLocal[3] = dUtils::AigNodeLitCond(rmMFFC[2], dUtils::AigNodeIsComplement(decGraphLocal[3]));

            }
       }
        else if(verbose>=2) printf("in resolving conflict rootId %d, remain %d <= %d\n", rootId, remain, decGraphLocal[0]);

    }
time2= clock();
printf("    takes %.2f sec to run\n", (time2 - time1) / (double) CLOCKS_PER_SEC);
    
    cudaMemcpy(dMaskValid, vMaskValid, sizeof(int)*nObjs, cudaMemcpyHostToDevice);
    cudaMemcpy(ddecGraph, decGraph, sizeof(int)*nObjs*DECGRAPH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dGain, vGain, sizeof(int)*nObjs, cudaMemcpyHostToDevice); 
    free(vCandRootId); free(vMaskValid); free(vIntMFFCNodes); free(vIntMFFCNodesInd); free(vMFFCNumSaved);
    // free(decGraph); 
    cudaFreeHost(decGraph); 
    free(vGain);
    free(solidDiv); free(travNodes);

time3 = clock();
printf("    takes %.2f sec to free\n", (time3 - time2) / (double) CLOCKS_PER_SEC);

printf("resolve conflict done, takes %.2f sec\n", (time3 - beginTime) / (double) CLOCKS_PER_SEC);
}


__global__ void printStatusDetail(int* d_pFanin0, int* d_pFanin1, int* d_pOuts, int nPIs, int nPOs, int nObjsNew, int* vMaskValid){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx==0) {
        for (int i = 0; i < nObjsNew; i++) {
            printf("%d(%d)\t", i, vMaskValid[i]);
            if (d_pFanin0[i] != -1 && i>nPIs)
                printf("%s%d\t", dUtils::AigNodeIsComplement(d_pFanin0[i]) ? "!" : "", dUtils::AigNodeID(d_pFanin0[i]));
            else
                printf("\t");
            if (d_pFanin1[i] != -1 && i>nPIs)
                printf("%s%d\t", dUtils::AigNodeIsComplement(d_pFanin1[i]) ? "!" : "", dUtils::AigNodeID(d_pFanin1[i]));
            else
                printf("\t");
            // printf("%d", d_pNumFanouts[i]);
            printf("\n");
        }
        for (int i = 0; i < nPOs; i++) {
            printf("%d\t", i + nObjsNew);
            printf("%s%d\n", dUtils::AigNodeIsComplement(d_pOuts[i]) ? "!" : "", dUtils::AigNodeID(d_pOuts[i]));
        }
    }
}


__global__ void danglingDetectorIter(int* d_pFanin0, int* d_pFanin1, int* d_pOuts, 
                int* vMaskValid, int iterRound, int* fanoutRef, int nNodes, int nObjs, int nPIs, int nPOs){
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < nObjs+nPOs; idx += nThreads) {
        int nodeId = idx;
        if(nodeId <= nPIs){
            fanoutRef[nodeId] = iterRound; 
            continue;
        }
        if(nodeId >= nObjs){
            fanoutRef[dUtils::AigNodeID(d_pOuts[nodeId-nObjs])] = iterRound;
            continue;
        }
        int faninId;
        if(vMaskValid[nodeId]!=STATUS_DELETE){
            faninId = dUtils::AigNodeID(d_pFanin0[nodeId]);
            fanoutRef[faninId] = iterRound;
            faninId = dUtils::AigNodeID(d_pFanin1[nodeId]);
            fanoutRef[faninId] = iterRound;
        }

    }
}


void danglingDetector(int* d_pFanin0, int* d_pFanin1, int* d_pOuts, int* vMaskValid, int nNodes, int nObjs, int nPIs, int nPOs, int verbose){
    int  nValid, nValidNew, nDeleted, nDeletedNew;
    int iterRound = 1;
    int* fanoutRef;
    cudaMalloc(&fanoutRef, sizeof(int)*nObjs);
    cudaMemset(fanoutRef, 0, sizeof(int)*nObjs);
    nDeleted = thrust::count(thrust::device, vMaskValid, vMaskValid+nObjs, STATUS_DELETE);
    do{
        nValid = nValidNew;
        danglingDetectorIter<<<NUM_BLOCKS(nObjs+nPOs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
           d_pFanin0, d_pFanin1, d_pOuts, vMaskValid, iterRound, fanoutRef, nNodes, nObjs, nPIs, nPOs
        );
        cudaDeviceSynchronize();

        nValidNew = thrust::count(thrust::device, fanoutRef, fanoutRef+nObjs, iterRound);
        thrust::transform_if(thrust::device, vMaskValid, vMaskValid+nObjs, fanoutRef, vMaskValid, unaryValue<STATUS_DELETE>() , dUtils::notEqualsValNC<int>(iterRound));

        iterRound ++;
    }while(nValid != nValidNew);
    nDeletedNew = thrust::count(thrust::device, vMaskValid, vMaskValid+nObjs, STATUS_DELETE);
    printf("detected %d dangling nodes\n", nDeletedNew - nDeleted);
    cudaFree(fanoutRef);

}



std::tuple<int, int *, int *, int *, int*> 
resubPerform(bool fUseZeros, bool fUseConstr, bool fUpdateLevel,
                    int cutSize, int addNodes,
                    int nObjs, int nPIs, int nPOs, int nNodes, 
                    int * d_pFanin0, int * d_pFanin1, int * d_pOuts, 
                    int * d_pNumFanouts, const int * d_pLevel, int verbose,
                    const int * pFanin0, const int * pFanin1, const int * pOuts){

    int  *vMFFCNumSaved, *vIntMFFCNodes, *vMFFCMark;
    clock_t time = clock();
printCheckPoint(0, time, "begin");

    // 1. data prepare
    cudaMalloc(&vMFFCNumSaved, sizeof(int) * nObjs);    // MFFC cone size
    cudaMemset(vMFFCNumSaved, 0, sizeof(int)*nObjs);
    cudaMalloc(&vMFFCMark, sizeof(int)*nObjs);  // store the nearest MFFC rootId of a node
    thrust::sequence(thrust::device, vMFFCMark, vMFFCMark+nObjs);
    thrust::fill(thrust::device, vMFFCMark, vMFFCMark+nPIs+1, 0);

    int *tempMffcCutSizes, *tempMffcTables;
    cudaMalloc(&tempMffcCutSizes, sizeof(int)*nObjs);
    cudaMalloc(&tempMffcTables, sizeof(int)*nObjs*CUT_TABLE_SIZE);
    getMffcNumResub<<<NUM_BLOCKS(nNodes, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pFanin0, d_pFanin1, d_pNumFanouts, d_pLevel, 
        tempMffcTables, tempMffcCutSizes, vMFFCNumSaved, nNodes, nPIs, cutSize*10
    );
    cudaDeviceSynchronize();
    cudaFree(tempMffcCutSizes);
    cudaFree(tempMffcTables);

    int* vIntMFFCNodesInd;
    cudaMalloc(&vIntMFFCNodesInd, sizeof(int)*nObjs);
    thrust::inclusive_scan(thrust::device, vMFFCNumSaved, vMFFCNumSaved+nObjs, vIntMFFCNodesInd);
    int nIntMFFCNodes;
    cudaMemcpy(&nIntMFFCNodes, &vIntMFFCNodesInd[nObjs - 1], sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA( cudaMalloc(&vIntMFFCNodes, sizeof(int)*nIntMFFCNodes) );

    int * vSeq;
    cudaMalloc(&vSeq, sizeof(int) * nObjs);
    thrust::sequence(thrust::device, vSeq, vSeq + nObjs);
    int* pNewGlobalListEnd;

    int * vMaskValid, * vValidEnumInd;
    cudaMalloc(&vMaskValid, sizeof(int) * nObjs);   
    cudaMalloc(&vValidEnumInd, sizeof(int) * nObjs);
    thrust::fill(thrust::device, vMaskValid, vMaskValid+nObjs, STATUS_VALID);

    int* decGraph;
    CHECK_CUDA( cudaMalloc(&decGraph, sizeof(int)*nObjs*DECGRAPH_SIZE) ); // for 3-resub, add 4 nodes at most
    cudaMemset(decGraph, 0, sizeof(int)*nObjs*DECGRAPH_SIZE);
    int* vGain;
    cudaMalloc(&vGain, sizeof(int)*nObjs);
    cudaMemset(vGain, 0, sizeof(int)*nObjs);

    int* vUsedNodes;    
    // 0 const; 1 replacement; 2 single or; 3 single and; 4 double or; 5 double and; 6 or-and; 7 and-or; 8 or-2and; 9 and-2or
    cudaMalloc(&vUsedNodes, sizeof(int)*nObjs);
    thrust::fill(thrust::device, vUsedNodes, vUsedNodes+nObjs, -1);
    
    // compute fanout array
    int * vFanoutRanges, *vFanoutInd;
    int nFanoutRange;
    CHECK_CUDA( cudaMalloc(&vFanoutRanges, sizeof(int) * nObjs) );
    thrust::inclusive_scan(thrust::device, d_pNumFanouts, d_pNumFanouts+nObjs, vFanoutRanges);
    CHECK_CUDA( cudaMemcpy(&nFanoutRange, &vFanoutRanges[nObjs - 1], sizeof(int), cudaMemcpyDeviceToHost) );
    assert(nFanoutRange == 2*nNodes+nPOs);
    cudaMalloc(&vFanoutInd, sizeof(int)*nFanoutRange);
    Aig::getFanout(pFanin0, pFanin1, pOuts, d_pNumFanouts, vFanoutRanges, vFanoutInd, nPIs, nObjs, nPOs, nFanoutRange);
    
    unsigned* vTruthElem;
    int nWords = dUtils::TruthWordNum(cutSize);
    cudaMalloc(&vTruthElem, sizeof(unsigned) * cutSize * nWords);
    Aig::getElemTruthTable<<<1, 1>>>(vTruthElem, cutSize);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // pepare reverse level and fanout
    int maxLevel, maxLevelSize;
    int *vLevelNodes, *vLevelNodesInd, *vReverseLevel;

    auto [l_maxLevel, l_maxLevelSize, l_vLevels, l_vLevelNodes, l_vLevelNodesInd, l_vMaxFanoutLevels] = Aig::reverseLevelWiseSchemeCPU(
        pFanin0, pFanin1, pOuts, nPIs, nObjs, nPOs
    );

    // copy vLevelNodes and vLevelRelease to device
    cudaMalloc(&vReverseLevel, sizeof(int)*nObjs);
    CHECK_CUDA( cudaMemcpy(vReverseLevel, l_vLevels, sizeof(int)*nObjs, cudaMemcpyHostToDevice) );
    // uncomment lines belows cause double free
    // free(l_vLevels);
    // free(l_vMaxFanoutLevels);

    // dynamically calculate the maxLevelUpperBound by memory
    int maxLevelUpperBound;
    size_t free_memory,t;
    cudaMemGetInfo(&free_memory, &t);
    // includes the three intermediate array in getMffcCutResub
    double scale = (double)(DIVISOR_SIZE * 3 + STACK_SIZE*7 + DIV2_MAX * 4 + nWords * DIVISOR_SIZE) * sizeof(int);
    maxLevelUpperBound = (int) ((double)free_memory * 0.85 / scale);

    int numNewLevel = (nObjs+maxLevelUpperBound-1) / maxLevelUpperBound;
    maxLevelSize = nObjs;
    vLevelNodes = vSeq;
    if(numNewLevel>l_maxLevel)
        vLevelNodesInd = (int*)realloc(l_vLevelNodesInd, sizeof(int)*numNewLevel);
    else
        vLevelNodesInd = l_vLevelNodesInd;
    assert(vLevelNodesInd);
    maxLevel = 0;
    while(maxLevelSize>maxLevelUpperBound){
        vLevelNodesInd[maxLevel] = (maxLevel+1)*maxLevelUpperBound;
        maxLevel++;
        maxLevelSize -= maxLevelUpperBound;
    }
    assert(maxLevel+1 == numNewLevel);
    maxLevelSize = maxLevel ? maxLevelUpperBound : maxLevelSize;
    vLevelNodesInd[maxLevel]= nObjs;
printf("  maxLevelSize: %d -> %d, maxLevel: %d -> %d\n", l_maxLevelSize, maxLevelSize, l_maxLevel, maxLevel);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    int * vMFFCTable, * vMFFCSizes;
    CHECK_CUDA( cudaMalloc(&vMFFCSizes, sizeof(int) * maxLevelSize) );   // MFFC cut size
    CHECK_CUDA( cudaMalloc(&vMFFCTable, sizeof(int) * maxLevelSize * CUT_TABLE_SIZE) );
    
    // alloc memory and prepare data
    int startIdx, endIdx, nLevelNodes;
    int * vCutTable, * vCutSizes;
    cudaMalloc(&vCutTable, sizeof(int) * maxLevelSize * CUT_TABLE_SIZE);
    cudaMalloc(&vCutSizes, sizeof(int) * maxLevelSize);
    cudaMemset(vCutSizes, 0, sizeof(int) * maxLevelSize);

    unsigned * vTruth;
    cudaMalloc(&vTruth, sizeof(unsigned) * maxLevelSize * nWords);
    // the internal nodes of TFI
    int* intNodesInd, *intNodesSize;
    cudaMalloc(&intNodesInd, sizeof(int) * STACK_SIZE * maxLevelSize);
    cudaMalloc(&intNodesSize, sizeof(int) * maxLevelSize);
    int* divisor, *divisorSize;   // divisor is just containing the index of divisors in intNodesInd and intNodesTruth
    CHECK_CUDA( cudaMalloc(&divisor, sizeof(int)*DIVISOR_SIZE*maxLevelSize) );
    cudaMalloc(&divisorSize, sizeof(int)*maxLevelSize);
    unsigned* divisorTruth;
printf("  divisorTruth needs %lu MB\n", sizeof(unsigned) * nWords * maxLevelSize * DIVISOR_SIZE/1024/1024) ;
    CHECK_CUDA( cudaMalloc(&divisorTruth, sizeof(unsigned) * nWords * maxLevelSize * DIVISOR_SIZE) );
    int* simSize;
    cudaMalloc(&simSize, sizeof(int)*maxLevelSize); // simSize = divSize + mffcSize, which is the #nodes to simulate
    int* nodePhase;
    cudaMalloc(&nodePhase, sizeof(int)*maxLevelSize*DIVISOR_SIZE);
    cudaMemset(nodePhase, 0, sizeof(int)*maxLevelSize*DIVISOR_SIZE);

    int* UP, * UN, * BI, *queueSize;
    CHECK_CUDA( cudaMalloc(&UP, sizeof(int)*maxLevelSize*STACK_SIZE) );
    cudaMalloc(&UN, sizeof(int)*maxLevelSize*STACK_SIZE);
    cudaMalloc(&BI, sizeof(int)*maxLevelSize*STACK_SIZE);
    cudaMalloc(&queueSize, sizeof(int)*maxLevelSize*5); // up1size, un1size, b1size, up2size, un2size
    int *UP20, *UP21, *UN20, *UN21;
    cudaMalloc(&UP20, sizeof(int)*maxLevelSize*DIV2_MAX);
    cudaMalloc(&UP21, sizeof(int)*maxLevelSize*DIV2_MAX);
    cudaMalloc(&UN20, sizeof(int)*maxLevelSize*DIV2_MAX);
    cudaMalloc(&UN21, sizeof(int)*maxLevelSize*DIV2_MAX);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
printCheckPoint(1, time, "data prepare");

    // 2. resub by reverse level (top-down)
    // reverse traverse
    for (int currLevel = 0; currLevel <= maxLevel; currLevel++) {
        startIdx = currLevel==0 ? 0 : vLevelNodesInd[currLevel - 1];
        endIdx = vLevelNodesInd[currLevel];
        nLevelNodes = endIdx - startIdx;
        assert(nLevelNodes <= maxLevelSize);
        if(verbose>=2)  printf("Reverse Level %d, nodes: %d\n", currLevel, nLevelNodes);

        int nValid =nLevelNodes;
        assert(nValid<=maxLevelSize);
        int* vValidEnumInd = vLevelNodes + startIdx;
        // collect cut
        getReconvCutResub<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            d_pFanin0, d_pFanin1, d_pNumFanouts, d_pLevel,
            vLevelNodes + startIdx, vCutTable, vCutSizes, nValid, nPIs, cutSize, 100,
            intNodesInd, intNodesSize
        );
        cudaDeviceSynchronize();

        if(verbose>=4){    
            printReconvCut<<<1,1>>>(
                d_pFanin0, d_pFanin1, d_pNumFanouts, d_pLevel,
                vLevelNodes + startIdx, vCutTable, vCutSizes, nValid, nPIs, cutSize, 100
            );
            cudaDeviceSynchronize();
        }
        
        // there is no need to set a cutSize for FFC
        getMffcCutResub<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            d_pFanin0, d_pFanin1, d_pNumFanouts, d_pLevel, 
            vLevelNodes + startIdx, vCutTable, vCutSizes, nValid, 
            vMFFCTable, vMFFCSizes, vMFFCNumSaved, 
            nNodes, nPIs, cutSize*10, vIntMFFCNodes, vIntMFFCNodesInd, vMFFCMark
        );
        cudaDeviceSynchronize();
        // collect div
        collectDivisor<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            d_pFanin0, d_pFanin1, d_pNumFanouts, d_pLevel, nPIs, nObjs, 
            vValidEnumInd, nValid, intNodesInd, intNodesSize, vCutTable, vCutSizes,
            vIntMFFCNodes, vIntMFFCNodesInd, vMFFCNumSaved, divisor, divisorSize, simSize,
            vFanoutRanges, vFanoutInd, 100, vMaskValid, vReverseLevel, vMFFCMark, 
            fUseConstr, fUpdateLevel, l_maxLevel
        );
        cudaDeviceSynchronize();

        simulateResub<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            d_pFanin0, d_pFanin1, vValidEnumInd, vCutTable, vCutSizes, 
            vTruth, vTruthElem, nValid, nPIs, cutSize, 
            divisor, divisorTruth, divisorSize, simSize, nodePhase
        );
        cudaDeviceSynchronize();

        // consier const
        resubDivsC<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vTruth, vValidEnumInd, nValid, vCutSizes, divisor, divisorTruth, divisorSize, simSize, cutSize,
            vMaskValid, vMFFCNumSaved, decGraph, nodePhase, vGain, verbose, vUsedNodes
        );
        cudaDeviceSynchronize();

        // consider equal nodes
        resubDivs0<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vTruth, vValidEnumInd, nValid, vCutSizes, divisor, divisorTruth, divisorSize, simSize, cutSize,
            vMaskValid, vMFFCNumSaved, decGraph, nodePhase, vGain, verbose, vUsedNodes
        );
        cudaDeviceSynchronize();

        if(addNodes<1)  continue;

        // consider one nodes
        resubDivs1<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vTruth, vValidEnumInd, nValid, vCutSizes, divisor, divisorTruth, divisorSize, simSize, cutSize, 
            vMaskValid, vMFFCNumSaved, decGraph, nodePhase, vGain, verbose, UP, UN, BI, queueSize, vUsedNodes
        );
        cudaDeviceSynchronize();

        if(addNodes<2)  continue;

        // consider triple
        resubDivs12<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vTruth, vValidEnumInd, nValid, vCutSizes, divisor, divisorTruth, divisorSize, simSize, cutSize, 
            vMaskValid, vMFFCNumSaved, decGraph, nodePhase, vGain, verbose, UP, UN, BI, queueSize, d_pLevel, vUsedNodes
        );
        cudaDeviceSynchronize();

        // consider two nodes
        resubDivs2<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vTruth, vValidEnumInd, nValid, vCutSizes, divisor, divisorTruth, divisorSize, simSize, cutSize, 
            vMaskValid, vMFFCNumSaved, decGraph, nodePhase, vGain, verbose, UP, UN, BI, UP20, UP21, UN20, UN21, queueSize, vUsedNodes
        );
        cudaDeviceSynchronize();

        if(addNodes<3)  continue;
        // consider three nodes
        resubDivs3<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
            vTruth, vValidEnumInd, nValid, vCutSizes, divisor, divisorTruth, divisorSize, simSize, cutSize, 
            vMaskValid, vMFFCNumSaved, decGraph, nodePhase, vGain, verbose, UP20, UP21, UN20, UN21, queueSize, vUsedNodes
        );
        cudaDeviceSynchronize();
    }

printCheckPoint(2, time, "collect");
    cudaFree(vCutTable); cudaFree(vCutSizes); 
    cudaFree(vMFFCTable); cudaFree(vMFFCSizes); 
    cudaFree(vReverseLevel);
    cudaFree(vFanoutRanges); cudaFree(vFanoutInd);
    cudaFree(vTruthElem); cudaFree(vTruth);
    cudaFree(intNodesInd); cudaFree(intNodesSize);
    cudaFree(divisor); cudaFree(divisorSize); cudaFree(divisorTruth); cudaFree(simSize);
    cudaFree(nodePhase);
    cudaFree(UP);   cudaFree(UN);   cudaFree(BI);   cudaFree(queueSize);
    cudaFree(UP20); cudaFree(UP21); cudaFree(UN20); cudaFree(UN21);
    free(vLevelNodesInd);


    // 3. pick the resub to commit
    // 3.1 filter nodes with pos gain
    int totalCandNum;
    int estGain, * vCandRootId;
    cudaMalloc(&vCandRootId, sizeof(int) * nObjs);
    pNewGlobalListEnd = thrust::copy_if(thrust::device, vSeq, vSeq + nObjs, vGain, vCandRootId, dUtils::greaterThanVal<int, 0>());
    int nCandRoot = pNewGlobalListEnd - vCandRootId;

    totalCandNum = nCandRoot;
    // 3.2 check conflict in parallel
    checkMffcConflict<<<NUM_BLOCKS(nCandRoot, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(vCandRootId, nCandRoot, vMaskValid, vIntMFFCNodes, vIntMFFCNodesInd, vMFFCNumSaved, vMFFCMark, decGraph, vGain, verbose);
    cudaDeviceSynchronize();
    cudaMemcpy(&estGain, vGain, sizeof(int), cudaMemcpyDeviceToHost);
    // 3.3 aggregate the nodes with conflict in its MFFC
    pNewGlobalListEnd = thrust::copy_if(thrust::device, vCandRootId, vCandRootId+nCandRoot, vCandRootId, dUtils::maskNotEqualsVal<int>(STATUS_ROOT, vMaskValid));
    nCandRoot = pNewGlobalListEnd - vCandRootId; 
    printf("total roots: %d = %d (no conflict, with gain %d) + %d (conflict)\n", 
            totalCandNum, totalCandNum-nCandRoot, estGain, nCandRoot);
    // 3.4 sort by gain and resolve conflict in sequential
    thrust::sort(thrust::device, vCandRootId, vCandRootId+nCandRoot, descSortBy(vGain));

    // if #cand is small, resolve them in GPU so that no need to copy memory to CPU
    if(fUseConstr && nCandRoot < 300000){
        int* solidDiv;
        cudaMalloc(&solidDiv, sizeof(int)*nObjs);
        cudaMemset(solidDiv, 0, sizeof(int)*nObjs);
        resolveMffcConflictGreedyGPU<<<1,1>>>(
                vCandRootId, nCandRoot, nObjs, nPIs, vMaskValid, 
                vIntMFFCNodes, vIntMFFCNodesInd, nIntMFFCNodes, vMFFCNumSaved, 
                decGraph, vGain, d_pFanin0, d_pFanin1, solidDiv, verbose, fUseConstr);
        cudaDeviceSynchronize();
        cudaFree(solidDiv);
    }
    else 
        resolveMffcConflictGreedy(vCandRootId, nCandRoot, nObjs, nPIs, vMaskValid, vIntMFFCNodes, vIntMFFCNodesInd, nIntMFFCNodes, vMFFCNumSaved, decGraph, vGain, pFanin0, pFanin1, verbose, fUseConstr);

    if(verbose>=2){
        printStatistics<<<1,1>>>(nPIs, nObjs, nNodes, addNodes, vGain, decGraph, vMaskValid, totalCandNum, nCandRoot, vUsedNodes);
        cudaDeviceSynchronize();
    }
printCheckPoint(3, time, "conflict resolve");

    // 4. reconstruct
    pNewGlobalListEnd = thrust::copy_if(thrust::device, vSeq, vSeq + nObjs, vMaskValid, vValidEnumInd, dUtils::isMinusOne<int>());
    int nValid = pNewGlobalListEnd - vValidEnumInd;
    printf("there are %d nodes to be replaced\n", nValid);
    assert(nNodes>=nPOs);
    reconstruct<<<NUM_BLOCKS(nNodes, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pFanin0, d_pFanin1, d_pOuts, nObjs, vSeq+1+nPIs, nNodes, nPOs, decGraph, vIntMFFCNodes, vIntMFFCNodesInd, vMFFCNumSaved, vMaskValid, d_pLevel
    );
    cudaDeviceSynchronize();
    // delete 0-resub and recover the inner nodes
    deleteZeroResub<<<NUM_BLOCKS(nValid, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(vValidEnumInd, nValid, decGraph, vMaskValid);
    cudaDeviceSynchronize();
    // attention that fanout has been changed
    danglingDetector(d_pFanin0, d_pFanin1, d_pOuts, vMaskValid,  nNodes, nObjs, nPIs, nPOs, verbose);
    // cudaDeviceSynchronize();
    int* oldToNewNodeId;
    cudaMalloc(&oldToNewNodeId, sizeof(int)*nObjs);
    thrust::transform(thrust::device, vMaskValid, vMaskValid+nObjs, oldToNewNodeId, ValidStatus<int>());
    thrust::inclusive_scan(thrust::device, oldToNewNodeId, oldToNewNodeId+nObjs, oldToNewNodeId);
    int nObjsNew;
    cudaMemcpy(&nObjsNew, &oldToNewNodeId[nObjs - 1], sizeof(int), cudaMemcpyDeviceToHost);
    printf("nNodes %d -> %d\n", nObjs-1-nPIs, nObjsNew-1-nPIs);
    int* vFanin0New, *vFanin1New;
    cudaMalloc(&vFanin0New, sizeof(int)*nObjsNew);
    cudaMalloc(&vFanin1New, sizeof(int)*nObjsNew);
    int* vOutNew;
    cudaMalloc(&vOutNew, sizeof(int)*nPOs);
    int* vNumFanouts;
    cudaMalloc(&vNumFanouts, sizeof(int)*nObjsNew);
    cudaMemset(vNumFanouts, 0, sizeof(int)*nObjsNew);

    remapNode<<<NUM_BLOCKS(nObjs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pFanin0, d_pFanin1, nPIs, nObjs, nObjsNew, oldToNewNodeId, vFanin0New, vFanin1New, vNumFanouts, vMaskValid
    );
    cudaDeviceSynchronize();
    
    remapPO<<<NUM_BLOCKS(nPOs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pOuts, vOutNew, vNumFanouts, nPOs, oldToNewNodeId, vMaskValid
    );
    cudaDeviceSynchronize();


    cudaFree(vMFFCNumSaved); cudaFree(vIntMFFCNodes); cudaFree(vIntMFFCNodesInd);   cudaFree(vMFFCMark);
    cudaFree(vSeq);
    cudaFree(vMaskValid); cudaFree(vValidEnumInd);
    cudaFree(decGraph);
    cudaFree(vGain);
    cudaFree(vUsedNodes);
    cudaFree(vCandRootId);
    cudaFree(oldToNewNodeId);
printCheckPoint(4, time, "reconstruct");


    return {nObjsNew, vFanin0New, vFanin1New, vOutNew, vNumFanouts};

}