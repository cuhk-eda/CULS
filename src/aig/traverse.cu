#include <vector>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include "traverse.cuh"


std::tuple<int, int, int *, int *, int *, int *>
Aig::levelWiseSchemeCPU(const int * vFanin0, const int * vFanin1, 
                        int nPIs, int nObjs, int * vLevels) {
    int maxLevel = 0, currLevel;
    int maxLevelSize = -1;
    int fanin0Id, fanin1Id;

    int * vMaxFanoutLevels; // the maximum fanout level of each node
    int * vLevelNodes;
    int * vLevelNodesInd;

    vMaxFanoutLevels = (int *) malloc(nObjs * sizeof(int));
    memset(vMaxFanoutLevels, -1, nObjs * sizeof(int));

    // compute (optional) vLevels and vMaxFanoutLevels
    if (vLevels == NULL) {
        // allocate and compute vLevels
        vLevels = (int *) malloc(nObjs * sizeof(int));
// FIXME: 
        for (int i = 0; i <= nPIs; i++)
            vLevels[i] = 0;
        for (int i = nPIs + 1; i < nObjs; i++) {
            fanin0Id = AigNodeID(vFanin0[i]), fanin1Id = AigNodeID(vFanin1[i]);
            currLevel = 1 + std::max(vLevels[fanin0Id], vLevels[fanin1Id]);
            vLevels[i] = currLevel;

            if (currLevel > maxLevel)
                maxLevel = currLevel;
            
            if (currLevel > vMaxFanoutLevels[fanin0Id])
                vMaxFanoutLevels[fanin0Id] = currLevel;
            if (currLevel > vMaxFanoutLevels[fanin1Id])
                vMaxFanoutLevels[fanin1Id] = currLevel;
        }
    } else {
        // get maxLevel
        for (int i = nPIs + 1; i < nObjs; i++) {
            currLevel = vLevels[i];

            if (currLevel > maxLevel)
                maxLevel = currLevel;
            
            if (currLevel > vMaxFanoutLevels[fanin0Id])
                vMaxFanoutLevels[fanin0Id] = currLevel;
            if (currLevel > vMaxFanoutLevels[fanin1Id])
                vMaxFanoutLevels[fanin1Id] = currLevel;
        }
    }

    std::vector<int> vLevelSize(maxLevel + 1, 0);
    std::vector<int> vLevelCount(maxLevel + 1, 0);
    vLevelNodes = (int *) malloc(nObjs * sizeof(int));
    vLevelNodesInd = (int *) malloc((maxLevel + 1) * sizeof(int));

    // compute the number of nodes in each level,
    // and update vMaxFanoutLevels if it is not assigned
    for (int i = 0; i < nObjs; i++) {
        vLevelSize[vLevels[i]]++;
        if (vMaxFanoutLevels[i] == -1)
            vMaxFanoutLevels[i] = vLevels[i];
    }

    // compute the prefix-sum of vLevelSize as vLevelNodesInd
    for (int i = 0; i <= maxLevel; i++) {
        vLevelNodesInd[i] = vLevelSize[i] + (i == 0 ? 0 : vLevelNodesInd[i - 1]);
        if (i > 0 && maxLevelSize < vLevelSize[i])
            maxLevelSize = vLevelSize[i];
    }
    assert(vLevelNodesInd[maxLevel] == nObjs);

    // compute vLevelNodes
    for (int i = 0; i < nObjs; i++) {
        int level = vLevels[i];
        int startIdx = (level == 0 ? 0 : vLevelNodesInd[level - 1]);

        vLevelNodes[startIdx + vLevelCount[level]++] = i;
    }

    return {maxLevel, maxLevelSize, vLevels, vLevelNodes, vLevelNodesInd, vMaxFanoutLevels};
}

std::tuple<int, int, int *, int *, int *, int *>
Aig::reverseLevelWiseSchemeCPU(const int * vFanin0, const int * vFanin1, const int * vOuts, 
                        int nPIs, int nObjs, int nPOs, int * vLevels) {
    int maxLevel = 0, currLevel;
    int maxLevelSize = -1;
    int fanin0Id, fanin1Id;

    int * vMaxFanoutLevels; // the maximum fanout level of each node
    int * vLevelNodes;
    int * vLevelNodesInd;

    vMaxFanoutLevels = (int *) malloc(nObjs * sizeof(int));
    memset(vMaxFanoutLevels, -1, nObjs * sizeof(int));

    if (vLevels == NULL) {
        // allocate and compute vLevels
        vLevels = (int *) calloc(nObjs , sizeof(int));
        for(int i=0; i<nPOs; i++)
            vLevels[AigNodeID(vOuts[i])] = 0;
        for(int i=nObjs-1; i>nPIs; i--){
            currLevel = vLevels[i];
            fanin0Id = AigNodeID(vFanin0[i]), fanin1Id = AigNodeID(vFanin1[i]);
            vLevels[fanin0Id] = max(vLevels[fanin0Id], currLevel+1);
            vLevels[fanin1Id] = max(vLevels[fanin1Id], currLevel+1);

        }

    } else {
        assert(0);
    }

    for(int i=0; i<=nPIs; i++)
        maxLevel = (maxLevel>vLevels[i]) ? maxLevel : vLevels[i];


    std::vector<int> vLevelSize(maxLevel + 1, 0);
    std::vector<int> vLevelCount(maxLevel + 1, 0);
    vLevelNodes = (int *) malloc(nObjs * sizeof(int));
    vLevelNodesInd = (int *) malloc((maxLevel + 1) * sizeof(int));

    for (int i = 0; i < nObjs; i++) {
        vLevelSize[vLevels[i]]++;
    }    
    for (int i = 0; i <= maxLevel; i++) {
        vLevelNodesInd[i] = vLevelSize[i] + (i == 0 ? 0 : vLevelNodesInd[i - 1]);
        // attention that in this case, the first reverse level (PO) is also considered
        // if (i > 0 && maxLevelSize < vLevelSize[i])
        if (maxLevelSize < vLevelSize[i])
            maxLevelSize = vLevelSize[i];
    }
    assert(vLevelNodesInd[maxLevel] == nObjs);

    for (int i = 0; i < nObjs; i++) {
        int level = vLevels[i];
        int startIdx = (level == 0 ? 0 : vLevelNodesInd[level - 1]);

        vLevelNodes[startIdx + vLevelCount[level]++] = i;
    }


    return {maxLevel, maxLevelSize, vLevels, vLevelNodes, vLevelNodesInd, vMaxFanoutLevels};

}

__global__ void getFanoutGPUSinglePO(const int* d_pOuts, int nPIs, int nObjs, int nPOs, 
        int* vFanoutRanges, int* vFanoutInd, int* fanoutRef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = gridDim.x * blockDim.x;
    int nodeId, startIdx, oldRef, fanin0Id;
    for(; idx<nPOs; idx+=nThreads){
        nodeId = idx+nObjs;
        fanin0Id = dUtils::AigNodeID(d_pOuts[idx]);
        startIdx = vFanoutRanges[fanin0Id];
        oldRef = atomicSub(&fanoutRef[fanin0Id], 1);
        assert(oldRef);
        vFanoutInd[startIdx-oldRef] = nodeId;

    }
}

// if vMarks[nodeId]=1, this node is blocked
// vMarks start from nNodes
__global__ void getFanoutGPUSingle(const int* d_pFanin0, const int* d_pFanin1, const int* d_pOuts, int nPIs, int nObjs, int nPOs, int* vFanoutRanges, int* vFanoutInd, int* fanoutRef, int* vMarks=NULL){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = gridDim.x * blockDim.x;
    int nodeId, startIdx, oldRef, fanin0Id, fanin1Id;
    for(; idx<nObjs; idx+=nThreads){
        nodeId = idx;
        // printf("nodeId %d, fanoutRef %d\n", nodeId, fanoutRef[nodeId]);
        if(dUtils::AigIsPIConst(nodeId, nPIs)) continue;
        if(vMarks!=NULL && vMarks[nodeId-nPIs-1]==1)   continue;

        fanin0Id = dUtils::AigNodeID(d_pFanin0[nodeId]);
        startIdx = vFanoutRanges[fanin0Id];
        oldRef = atomicSub(&fanoutRef[fanin0Id], 1);
        assert(oldRef);
        vFanoutInd[startIdx-oldRef] = nodeId;

        fanin1Id = dUtils::AigNodeID(d_pFanin1[nodeId]);
        startIdx = vFanoutRanges[fanin1Id];
        oldRef = atomicSub(&fanoutRef[fanin1Id], 1);
        assert(oldRef);
        vFanoutInd[startIdx-oldRef] = nodeId;

    }
}


void Aig::getFanoutGPU(const int* d_pFanin0, const int* d_pFanin1, const int* d_pOuts, const int* d_pNumFanouts, int* vFanoutRanges, int* vFanoutInd, int nPIs, int nObjs, int nPOs, int* vMarks){
    int *fanoutRef;
    cudaMalloc(&fanoutRef, sizeof(int)*nObjs);
    cudaMemcpy(fanoutRef, d_pNumFanouts, sizeof(int)*nObjs, cudaMemcpyDeviceToDevice);
    getFanoutGPUSinglePO<<<NUM_BLOCKS(nPOs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pOuts, nPIs, nObjs, nPOs, vFanoutRanges, vFanoutInd, fanoutRef);
    cudaDeviceSynchronize();
    getFanoutGPUSingle<<<NUM_BLOCKS(nObjs, THREAD_PER_BLOCK), THREAD_PER_BLOCK>>>(
        d_pFanin0, d_pFanin1, d_pOuts, nPIs, nObjs, nPOs, 
        vFanoutRanges, vFanoutInd, fanoutRef, vMarks);
    cudaDeviceSynchronize();
    
    cudaFree(fanoutRef);

}

void Aig::getFanout(const int* pFanin0, const int* pFanin1, const int* pOuts, const int* d_pNumFanouts, int* vFanoutRanges, int* vFanoutInd, 
                        int nPIs, int nObjs, int nPOs, int nFanoutRange){
    int *fanoutRef = (int*)malloc(sizeof(int)*nObjs);
    cudaMemcpy(fanoutRef, d_pNumFanouts, sizeof(int)*nObjs, cudaMemcpyDeviceToHost);
    int* pFanoutRanges = (int*)malloc(sizeof(int)*nObjs);
    cudaMemcpy(pFanoutRanges, vFanoutRanges, sizeof(int)*nObjs, cudaMemcpyDeviceToHost);
    int* pFanoutInd = (int*)malloc(sizeof(int)*nFanoutRange);
    int nodeId, startIdx, oldRef, fanin0Id, fanin1Id;
    for(int idx=nPOs-1; idx>=0; idx--){
        nodeId = idx+nObjs;
        fanin0Id = pOuts[idx]>>1;
        startIdx = pFanoutRanges[fanin0Id];
        oldRef = fanoutRef[fanin0Id]--;
        assert(oldRef);
        pFanoutInd[startIdx-oldRef] = nodeId;
    }

    for(int idx=nObjs-1; idx>nPIs; idx--){
        nodeId = idx;
        fanin0Id = pFanin0[nodeId] >> 1;
        startIdx = pFanoutRanges[fanin0Id];
        oldRef = fanoutRef[fanin0Id] --;
        assert(oldRef);
        pFanoutInd[startIdx-oldRef] = nodeId;

        fanin1Id = pFanin1[nodeId] >> 1;
        startIdx = pFanoutRanges[fanin1Id];
        oldRef = fanoutRef[fanin1Id] --;
        assert(oldRef);
        pFanoutInd[startIdx-oldRef] = nodeId;
    }

    cudaMemcpy(vFanoutInd, pFanoutInd, sizeof(int)*nFanoutRange, cudaMemcpyHostToDevice);

    free(fanoutRef);
    free(pFanoutRanges);
    free(pFanoutInd);

}
