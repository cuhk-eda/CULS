#include <tuple>
#include "common.h"
#include "resub.h"
#include "aig/truth.cuh"
#include "aig/strash.cuh"
#include "aig/traverse.cuh"

// 0-resub
__device__ void buildDecGraph(int* dec, int rootLit, int obj0Lit){
    dec[0] = 0;
    dec[1] = dUtils::AigNodeNotCond(obj0Lit, dUtils::AigNodeIsComplement(rootLit));
// printf("building decGraph %s%d\n", (dec[1]&1)?"!":" ", dec[1]>>1);
}

// 1-resub
__device__ void buildDecGraph(int* dec, int rootLit, int obj0Lit, int obj1Lit, int orGate){
    dec[0] = 1;
    dec[1] = dUtils::AigNodeNotCond(rootLit, orGate);
    dec[2] = dUtils::AigNodeNotCond(obj0Lit, orGate);
    dec[3] = dUtils::AigNodeNotCond(obj1Lit, orGate);
// printf("building decGraph %s%d = %s%d %s%d\n", dec[1]&1?"!":" ", dec[1]>>1, dec[2]&1?"!":" ", dec[2]>>1, dec[3]&1?"!":" ", dec[3]>>1  );
}

// 2-resub rootLit -> nodeLIt, obj2Lit; nodeLit-> obj0Lit, obj1Lit
__device__ void buildDecGraph(int* dec, int rootLit, int obj0Lit, int obj1Lit, int obj2Lit, int orGate){
    dec[0] = 2;
    dec[1] = dUtils::AigNodeNotCond(rootLit, orGate);
    dec[2] = dUtils::AigNodeNotCond(0, 0);
    dec[3] = dUtils::AigNodeNotCond(obj0Lit, orGate);
    dec[4] = dUtils::AigNodeNotCond(obj1Lit, orGate);
    dec[5] = dUtils::AigNodeNotCond(obj2Lit, orGate);
// printf("building decGraph %s%d =  %s%d %s%d %s%d orGate: %d\n", dec[1]&1?"!":" ", dec[1]>>1, dec[3]&1?"!":" ", dec[3]>>1, dec[4]&1?"!":" ", dec[4]>>1, dec[5]&1?"!":" ", dec[5]>>1, orGate);
}

__device__ void buildDecGraph22(int* dec, int rootLit, int obj0Lit, int obj1Lit, int obj2Lit, int orGate){
    // orGate = 1: (a & b) | c
    // orGate = 0: (a | b) & c
    dec[0] = 2;
    dec[1] = dUtils::AigNodeNotCond(rootLit, orGate==1);
    dec[2] = dUtils::AigNodeNotCond(0, 1);
    dec[3] = dUtils::AigNodeNotCond(obj0Lit, orGate==0);
    dec[4] = dUtils::AigNodeNotCond(obj1Lit, orGate==0);
    dec[5] = dUtils::AigNodeNotCond(obj2Lit, orGate==1);
// printf("building decGraph2 %s%d =  %s%d %s%d %s%d orGate: %d\n", dec[1]&1?"!":" ", dec[1]>>1, dec[3]&1?"!":" ", dec[3]>>1, dec[4]&1?"!":" ", dec[4]>>1, dec[5]&1?"!":" ", dec[5]>>1, orGate);
}

__device__ void buildDecGraph(int* dec, int rootLit, int obj0Lit, int obj1Lit, int obj2Lit, int obj3Lit, int orGate){
    // orGate = 1: (a & b) | (c & d)
    // orGate = 0: (a | b) & (c | d)
    dec[0] = 3;
    dec[1] = dUtils::AigNodeNotCond(rootLit, orGate==1);
    dec[2] = dUtils::AigNodeNotCond(0, 1);
    dec[3] = dUtils::AigNodeNotCond(0, 1);

    dec[4] = dUtils::AigNodeNotCond(obj0Lit, orGate==0);
    dec[5] = dUtils::AigNodeNotCond(obj1Lit, orGate==0);
    dec[6] = dUtils::AigNodeNotCond(obj2Lit, orGate==0);
    dec[7] = dUtils::AigNodeNotCond(obj3Lit, orGate==0);
// printf("building decGraph %s%d =  %s%d %s%d %s%d orGate: %d\n", dec[1]&1?"!":" ", dec[1]>>1, dec[3]&1?"!":" ", dec[3]>>1, dec[4]&1?"!":" ", dec[4]>>1, dec[5]&1?"!":" ", dec[5]>>1, orGate);
}


__global__ void resubDivsC(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize, 
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* vUsedNodes){

    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId;
    int nWords, nWordsElem = dUtils::TruthWordNum(nMaxCutSize);
    unsigned *divTruthLocal;
    // unsigned *rootTruth = (unsigned*)malloc(sizeof(unsigned)*nWordsElem);
    unsigned *puRoot;
    int* nodePhaseLocal;
    

    for (; idx < nValid; idx += nThreads) {
        rootId = vValidEnumInd[idx];
        assert(vMaskValid[rootId]==STATUS_VALID || vMaskValid[rootId]==STATUS_SKIP);
        if(vMFFCNumSaved[rootId]==0)    continue;
        int rootIdx = simSize[idx]-1;
        divTruthLocal = divisorTruth + (long unsigned int)idx*DIVISOR_SIZE*nWordsElem;
        int nVars = vCutSize[idx];
        nodePhaseLocal = nodePhase + idx*DIVISOR_SIZE;
        nWords = dUtils::TruthWordNum(nVars);
        puRoot = vTruth+idx*nWordsElem;
        assert(*(divTruthLocal+rootIdx*nWords) ^ *(puRoot)==0);
        int w;
        BreakWhen(w, 0, nWords, puRoot[w]);
        if(w==nWords){
            int gain = vMFFCNumSaved[rootId] ;
            vGain[rootId] = gain;
            vUsedNodes[rootId] = 0;
            buildDecGraph(decGraph+rootId*DECGRAPH_SIZE, 
                            dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]), 
                            1);
            if(verbose>=3)    printf("find C-resub %s%d <- 0, gain: %d\n", (nodePhaseLocal[rootIdx])?"!":" ", rootId, gain);
        }

    }                            
    
}

__global__ void resubDivs0(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize, 
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* vUsedNodes){

    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId;
    int* divisorLocal;
    int nWords, nWordsElem = dUtils::TruthWordNum(nMaxCutSize);
    unsigned *divTruthLocal;
    unsigned *puRoot, *pu0;
    int* nodePhaseLocal;
    

    for (; idx < nValid; idx += nThreads) {
        rootId = vValidEnumInd[idx];
        // DEBUGGING
        assert(vMaskValid[rootId]==STATUS_VALID || vMaskValid[rootId]==STATUS_SKIP);
        // if(vMaskValid[rootId]!=STATUS_VALID)   continue;
        if(vMFFCNumSaved[rootId]==0)    continue;
        if(vGain[rootId]) continue;   // skip the node has found a C-resub
        int rootIdx = simSize[idx]-1;
        int divSize = divisorSize[idx];
        divisorLocal = divisor + idx * DIVISOR_SIZE;
        divTruthLocal = divisorTruth + (long unsigned int)idx*DIVISOR_SIZE*nWordsElem;
        int nVars = vCutSize[idx];
        nodePhaseLocal = nodePhase + idx*DIVISOR_SIZE;
        nWords = dUtils::TruthWordNum(nVars);
        puRoot = vTruth+idx*nWordsElem;
        assert(*(divTruthLocal+rootIdx*nWords) ^ *(puRoot)==0);
        for(int i=0; i<divSize; i++){
            int candId = divisorLocal[i];
            int w;
            pu0 = divTruthLocal+i*nWords;
            BreakWhen(w, 0, nWords, puRoot[w] ^ pu0[w]);
            if( w == nWords){
                int gain = vMFFCNumSaved[rootId] ;
                vGain[rootId] = gain;
                vUsedNodes[rootId] = 1;
                buildDecGraph(decGraph+rootId*DECGRAPH_SIZE, 
                                dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]), 
                                dUtils::AigNodeLitCond(candId, nodePhaseLocal[i]));
                vMaskValid[candId] = STATUS_SKIP;
                if(verbose>=3)    printf("find 0-resub %s%d <- %s%d, gain: %d\n", (nodePhaseLocal[rootIdx])?"!":" ", rootId, 
                                                                (nodePhaseLocal[i])?"!":" ", candId, gain);

            }
        }
        
    }                            
    
}

__global__ void resubDivs1(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize,
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* UPGlobal, int* UNGlobal, int* BIGlobal, int* queueSize, int* vUsedNodes){

    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId;
    int * divisorLocal;
    int w, nWords, nWordsElem = dUtils::TruthWordNum(nMaxCutSize);
    unsigned *divTruthLocal;
    unsigned *puRoot;
    int* nodePhaseLocal;

    // store the idx of divisor; attention it has polarity
    int* UP1, *UN1, *B1;
    int up1Size, un1Size, b1Size;
    bool terminate;
    

    for (; idx < nValid; idx += nThreads) {
        rootId = vValidEnumInd[idx];
        UP1 = UPGlobal + idx*STACK_SIZE;
        UN1 = UNGlobal + idx*STACK_SIZE;
        B1 = BIGlobal + idx*STACK_SIZE;
        assert(vMaskValid[rootId]==STATUS_VALID || vMaskValid[rootId]==STATUS_SKIP);
        if(vGain[rootId]) continue;   // skip the node has found a 0-resub
        int rootIdx = simSize[idx]-1;
        int divSize = divisorSize[idx];
        int MFFCSize = vMFFCNumSaved[rootId];
        if(MFFCSize<=1)  continue;
        divisorLocal = divisor + idx * DIVISOR_SIZE;
        divTruthLocal = divisorTruth + (long unsigned int)idx*DIVISOR_SIZE*nWordsElem;
        int nVars = vCutSize[idx];
        nodePhaseLocal = nodePhase + idx*DIVISOR_SIZE;
        terminate = false;
        nWords = dUtils::TruthWordNum(nVars);
        puRoot = vTruth+idx*nWordsElem;
        up1Size = un1Size = b1Size = 0;

        // check pos/neg containment
        for(int i=0; i<divSize; i++){
            unsigned *pu0 = divTruthLocal+i*nWords;
            BreakWhen(w, 0, nWords, pu0[w] & ~puRoot[w]);
            if(w==nWords){  UP1[up1Size++] = dUtils::AigNodeLitCond(i, 0); continue;}
            BreakWhen(w, 0, nWords, ~pu0[w] & ~puRoot[w]);
            if(w==nWords){  
                for(w=0; w<nWords; w++)
                    pu0[w] = ~pu0[w];
                UP1[up1Size++] = dUtils::AigNodeLitCond(i, 1); 
                continue;
            }
            BreakWhen(w, 0, nWords, ~pu0[w] & puRoot[w]);
            if(w==nWords){  UN1[un1Size++] = dUtils::AigNodeLitCond(i, 0); continue;}
            BreakWhen(w, 0, nWords, pu0[w] & puRoot[w]);
            if(w==nWords){  
                for(w=0; w<nWords; w++)
                    pu0[w] = ~pu0[w];
                UN1[un1Size++] = dUtils::AigNodeLitCond(i, 1); 
                continue;
            }
            B1[b1Size++] = i;
        }
        queueSize[idx*5] = up1Size;
        queueSize[idx*5+1] = un1Size;
        queueSize[idx*5+2]= b1Size;

        // check the pos divisor with or gate
        for(int i=0; i<up1Size; i++){
            unsigned *pu0 = divTruthLocal+ dUtils::AigNodeID(UP1[i]) *nWords;
            for(int j=i+1; j<up1Size; j++){
                unsigned *pu1 = divTruthLocal+dUtils::AigNodeID(UP1[j]) *nWords;
                BreakWhen(w, 0, nWords, (pu0[w] | pu1[w]) ^ puRoot[w]);
                if(w==nWords){
                    int gain = MFFCSize -1;
                    vGain[rootId] = MFFCSize - 1;
                    // atomicAdd(vGain, gain);
                    vUsedNodes[rootId] = 2;
                    buildDecGraph(decGraph+rootId*DECGRAPH_SIZE, 
                                    dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]), 
                                    dUtils::AigNodeLitCond(divisorLocal[UP1[i]>>1], nodePhaseLocal[UP1[i]>>1]^dUtils::AigNodeIsComplement(UP1[i])), 
                                    dUtils::AigNodeLitCond(divisorLocal[UP1[j]>>1], nodePhaseLocal[UP1[j]>>1]^dUtils::AigNodeIsComplement(UP1[j])), 
                                    1
                    );
                    terminate = true;
                    assert(vMaskValid[divisorLocal[UP1[i]>>1]]!=STATUS_ROOT);
                    assert(vMaskValid[divisorLocal[UP1[j]>>1]]!=STATUS_ROOT);
                    vMaskValid[divisorLocal[UP1[i]>>1]] = STATUS_SKIP;
                    vMaskValid[divisorLocal[UP1[j]>>1]] = STATUS_SKIP;
                    if(verbose>=3)    printf("find 1-resub %d <- %s%d or %s%d, gain: %d\n", rootId, dUtils::AigNodeIsComplement(UP1[i])?"!":" ", divisorLocal[UP1[i]>>1], 
                                            dUtils::AigNodeIsComplement(UP1[j])?"!":" " ,divisorLocal[UP1[j]>>1], gain);

                }
                if(terminate)   break;
            }
            if(terminate)   break;
        }
        if(terminate)   continue;

        // check the neg divisor with and gate
        for(int i=0; i<un1Size; i++){
            unsigned *pu0 = divTruthLocal+ dUtils::AigNodeID(UN1[i]) *nWords;
            for(int j=i+1; j<un1Size; j++){
                unsigned *pu1 = divTruthLocal+dUtils::AigNodeID(UN1[j]) *nWords;
                BreakWhen(w, 0, nWords, (pu0[w] & pu1[w]) ^ puRoot[w]);
                if(w==nWords){
                    int gain = MFFCSize -1;
                    vGain[rootId] = MFFCSize - 1;
                    // atomicAdd(vGain, gain);
                    vUsedNodes[rootId] = 3;
                    buildDecGraph(decGraph+rootId*DECGRAPH_SIZE, 
                                    dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]),  
                                    dUtils::AigNodeLitCond(divisorLocal[UN1[i]>>1], nodePhaseLocal[UN1[i]>>1]^dUtils::AigNodeIsComplement(UN1[i])), 
                                    dUtils::AigNodeLitCond(divisorLocal[UN1[j]>>1], nodePhaseLocal[UN1[j]>>1]^dUtils::AigNodeIsComplement(UN1[j])), 
                                    0
                    );
                    terminate = true;
                    assert(vMaskValid[divisorLocal[UN1[i]>>1]]!=STATUS_ROOT);
                    assert(vMaskValid[divisorLocal[UN1[j]>>1]]!=STATUS_ROOT);
                    vMaskValid[divisorLocal[UN1[i]>>1]] = STATUS_SKIP;
                    vMaskValid[divisorLocal[UN1[j]>>1]] = STATUS_SKIP;
                    if(verbose>=3)    printf("find 1-resub %d <- %s%d and %s%d, gain: %d\n", rootId, 
                                            dUtils::AigNodeIsComplement(UN1[i])?"!":" ", divisorLocal[UN1[i]>>1], 
                                            dUtils::AigNodeIsComplement(UN1[j])?"!":" " ,divisorLocal[UN1[j]>>1], gain);
                }
                if(terminate)   break;
            }
            if(terminate)   break;
        }

        
    }                            
    
}


__global__ void resubDivs12(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize,
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* UPGlobal, int* UNGlobal, int* BIGlobal, int* queueSize, const int* pLevel, int* vUsedNodes){

    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId;
    int * divisorLocal;
    int w, nWords, nWordsElem = dUtils::TruthWordNum(nMaxCutSize);
    unsigned *divTruthLocal;
    unsigned *puRoot;
    int* nodePhaseLocal;

    // store the idx of divisor; attention it has polarity
    int* UP1, *UN1;
    int up1Size, un1Size;
    bool terminate;
 
    for (; idx < nValid; idx += nThreads) {
        rootId = vValidEnumInd[idx];
        UP1 = UPGlobal + idx*STACK_SIZE;
        UN1 = UNGlobal + idx*STACK_SIZE;
        assert(vMaskValid[rootId]==STATUS_VALID || vMaskValid[rootId]==STATUS_SKIP);
        if(vGain[rootId]) continue;   // skip the node has found a resub
        int rootIdx = simSize[idx]-1;
        int MFFCSize = vMFFCNumSaved[rootId];
        if(MFFCSize<=2)  continue;
        divisorLocal = divisor + idx * DIVISOR_SIZE;
        divTruthLocal = divisorTruth + (long unsigned int)idx*DIVISOR_SIZE*nWordsElem;
        int nVars = vCutSize[idx];
        nodePhaseLocal = nodePhase + idx*DIVISOR_SIZE;
        terminate = false;
        nWords = dUtils::TruthWordNum(nVars);
        puRoot =  vTruth+idx*nWordsElem;
        up1Size = queueSize[idx*5];
        un1Size = queueSize[idx*5+1];

        // check the pos divisor with or gate
        for(int i=0; i<up1Size; i++){
            unsigned *pu0 = divTruthLocal+ dUtils::AigNodeID(UP1[i]) *nWords;
            for(int j=i+1; j<up1Size; j++){
                unsigned *pu1 = divTruthLocal+dUtils::AigNodeID(UP1[j]) *nWords;
                for(int k=j+1; k<up1Size; k++){
                    unsigned *pu2 = divTruthLocal+dUtils::AigNodeID(UP1[k]) *nWords;
                    BreakWhen(w, 0, nWords, (pu0[w] | pu1[w] | pu2[w]) ^ puRoot[w]);
                    if(w==nWords){
                        int gain = MFFCSize -2;
                        vGain[rootId] = gain;
                        vUsedNodes[rootId] = 4;
                        int levelMax = max( max(pLevel[divisorLocal[UP1[i]>>1]], pLevel[divisorLocal[UP1[j]>>1]]), pLevel[divisorLocal[UP1[k]>>1]] );
                        int objMax, objMin0, objMin1;
                        if(levelMax==pLevel[divisorLocal[UP1[i]>>1]]){
                            objMax = UP1[i];    objMin0 = UP1[j];   objMin1 = UP1[k];
                        }
                        else if(levelMax==pLevel[divisorLocal[UP1[j]>>1]]){
                            objMax = UP1[j];    objMin0 = UP1[i];   objMin1 = UP1[k];
                        }
                        else if(levelMax==pLevel[divisorLocal[UP1[k]>>1]]){
                            objMax = UP1[k];    objMin0 = UP1[i];   objMin1 = UP1[j];
                        }


                        buildDecGraph(decGraph+rootId*DECGRAPH_SIZE, 
                                        dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]), 
                                        dUtils::AigNodeLitCond(divisorLocal[objMin0>>1], nodePhaseLocal[objMin0>>1]^dUtils::AigNodeIsComplement(objMin0)), 
                                        dUtils::AigNodeLitCond(divisorLocal[objMin1>>1], nodePhaseLocal[objMin1>>1]^dUtils::AigNodeIsComplement(objMin1)), 
                                        dUtils::AigNodeLitCond(divisorLocal[objMax>>1], nodePhaseLocal[objMax>>1]^dUtils::AigNodeIsComplement(objMax)), 
                                        1
                        );
                        terminate = true;
                        assert(vMaskValid[divisorLocal[UP1[i]>>1]]!=STATUS_ROOT);
                        assert(vMaskValid[divisorLocal[UP1[j]>>1]]!=STATUS_ROOT);
                        assert(vMaskValid[divisorLocal[UP1[k]>>1]]!=STATUS_ROOT);
                        vMaskValid[divisorLocal[UP1[i]>>1]] = STATUS_SKIP;
                        vMaskValid[divisorLocal[UP1[j]>>1]] = STATUS_SKIP;
                        vMaskValid[divisorLocal[UP1[k]>>1]] = STATUS_SKIP;
                        if(verbose>=3)    printf("find 21-resub %d <- %s%d or %s%d or %s%d, gain: %d\n", rootId, 
                                                dUtils::AigNodeIsComplement(UP1[i])?"!":" ", divisorLocal[UP1[i]>>1], 
                                                dUtils::AigNodeIsComplement(UP1[j])?"!":" " ,divisorLocal[UP1[j]>>1],
                                                dUtils::AigNodeIsComplement(UP1[k])?"!":" " ,divisorLocal[UP1[k]>>1],
                                                gain);

                    }
                    if(terminate)   break;
                }
                if(terminate)   break;
            }
            if(terminate)   break;
        }
        if(terminate)   continue;

        // check the neg divisor with and gate
        for(int i=0; i<un1Size; i++){
            unsigned *pu0 = divTruthLocal+ dUtils::AigNodeID(UN1[i]) *nWords;
            for(int j=i+1; j<un1Size; j++){
                unsigned *pu1 = divTruthLocal+dUtils::AigNodeID(UN1[j]) *nWords;
                for(int k=j+1; k<un1Size; k++){
                    unsigned *pu2 = divTruthLocal+dUtils::AigNodeID(UN1[k]) *nWords;
                    BreakWhen(w, 0, nWords, (pu0[w] & pu1[w] & pu2[w]) ^ puRoot[w]);
                    if(w==nWords){
                        int gain = MFFCSize -2;
                        vGain[rootId] = gain;
                        vUsedNodes[rootId] = 5;
                        int levelMax = max( max(pLevel[divisorLocal[UN1[i]>>1]], pLevel[divisorLocal[UN1[j]>>1]]), pLevel[divisorLocal[UN1[k]>>1]] );
                        int objMax, objMin0, objMin1;
                        if(levelMax==pLevel[divisorLocal[UN1[i]>>1]]){
                            objMax = UN1[i];    objMin0 = UN1[j];   objMin1 = UN1[k];
                        }
                        else if(levelMax==pLevel[divisorLocal[UN1[j]>>1]]){
                            objMax = UN1[j];    objMin0 = UN1[i];   objMin1 = UN1[k];
                        }
                        else if(levelMax==pLevel[divisorLocal[UN1[k]>>1]]){
                            objMax = UN1[k];    objMin0 = UN1[i];   objMin1 = UN1[j];
                        }


                        buildDecGraph(decGraph+rootId*DECGRAPH_SIZE, 
                                        dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]), 
                                        dUtils::AigNodeLitCond(divisorLocal[objMin0>>1], nodePhaseLocal[objMin0>>1]^dUtils::AigNodeIsComplement(objMin0)), 
                                        dUtils::AigNodeLitCond(divisorLocal[objMin1>>1], nodePhaseLocal[objMin1>>1]^dUtils::AigNodeIsComplement(objMin1)), 
                                        dUtils::AigNodeLitCond(divisorLocal[objMax>>1], nodePhaseLocal[objMax>>1]^dUtils::AigNodeIsComplement(objMax)), 
                                        0
                        );
                        terminate = true;
                        assert(vMaskValid[divisorLocal[UN1[i]>>1]]!=STATUS_ROOT);
                        assert(vMaskValid[divisorLocal[UN1[j]>>1]]!=STATUS_ROOT);
                        assert(vMaskValid[divisorLocal[UN1[k]>>1]]!=STATUS_ROOT);
                        vMaskValid[divisorLocal[UN1[i]>>1]] = STATUS_SKIP;
                        vMaskValid[divisorLocal[UN1[j]>>1]] = STATUS_SKIP;
                        vMaskValid[divisorLocal[UN1[k]>>1]] = STATUS_SKIP;
                        if(verbose>=3)    printf("find 21-resub %d <- %s%d and %s%d and %s%d, gain: %d\n", rootId, 
                                                dUtils::AigNodeIsComplement(UN1[i])?"!":" ", divisorLocal[UN1[i]>>1], 
                                                dUtils::AigNodeIsComplement(UN1[j])?"!":" " ,divisorLocal[UN1[j]>>1],
                                                dUtils::AigNodeIsComplement(UN1[k])?"!":" " ,divisorLocal[UN1[k]>>1],
                                                gain);
                    }
                    if(terminate)   break;
                }
                if(terminate)   break;
            }
            if(terminate)   break;
        }
 
    }                            

}



__global__ void resubDivs2(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize,
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose, int* UPGlobal, int* UNGlobal, int* BIGlobal,
                            int* UP20Global, int* UP21Global, int* UN20Global, int* UN21Global,  
                            int* queueSize, int* vUsedNodes){

    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId;
    int * divisorLocal;
    int w, nWords, nWordsElem = dUtils::TruthWordNum(nMaxCutSize);
    unsigned *divTruthLocal;
    unsigned* puRoot;
    int* nodePhaseLocal;

    // store the idx of divisor; attention it has polarity
    int* UP1, *UN1, *B1;
    int up1Size, un1Size, b1Size;
    bool terminate;
    int *UP20, *UP21, *UN20, *UN21;
    int up2Size=0, un2Size=0;
 
    for (; idx < nValid; idx += nThreads) {
        rootId = vValidEnumInd[idx];
        UP1 = UPGlobal + idx*STACK_SIZE;
        UN1 = UNGlobal + idx*STACK_SIZE;
        B1 = BIGlobal + idx*STACK_SIZE;
        UP20 = UP20Global + idx*DIV2_MAX;
        UP21 = UP21Global + idx*DIV2_MAX;
        UN20 = UN20Global + idx*DIV2_MAX;
        UN21 = UN21Global + idx*DIV2_MAX;
        assert(vMaskValid[rootId]==STATUS_VALID || vMaskValid[rootId]==STATUS_SKIP);
        // if(vMaskValid[rootId]!=STATUS_VALID)   continue;
        if(vGain[rootId]) continue;   // skip the node has found a resub
        int rootIdx = simSize[idx]-1;
        // int divSize = divisorSize[idx];
        int MFFCSize = vMFFCNumSaved[rootId];
        if(MFFCSize<=2)  continue;
        divisorLocal = divisor + idx * DIVISOR_SIZE;
        divTruthLocal = divisorTruth + (long unsigned int)idx*DIVISOR_SIZE*nWordsElem;
        int nVars = vCutSize[idx];
        nodePhaseLocal = nodePhase + idx*DIVISOR_SIZE;
        terminate = false;
        nWords = dUtils::TruthWordNum(nVars);
        puRoot = vTruth+idx*nWordsElem;
        up1Size = queueSize[idx*5];
        un1Size = queueSize[idx*5+1];
        b1Size = queueSize[idx*5+2];

        // collect double nodes divisors
        up2Size = un2Size = 0;
        for(int i=0; i<b1Size; i++){
            unsigned *pu0 = divTruthLocal + B1[i] *nWords;
            for(int j=i+1; j<b1Size; j++){
                unsigned *pu1 = divTruthLocal + B1[j]*nWords;
                if(up2Size<DIV2_MAX-4){
                    BreakWhen(w, 0, nWords, ( pu0[w]& pu1[w]) & ~puRoot[w]);
                    if(w==nWords){UP20[up2Size] = dUtils::AigNodeLitCond(B1[i], 0); UP21[up2Size++] = dUtils::AigNodeLitCond(B1[j], 0); }
                    BreakWhen(w, 0, nWords, (~pu0[w]& pu1[w]) & ~puRoot[w]);
                    if(w==nWords){UP20[up2Size] = dUtils::AigNodeLitCond(B1[i], 1); UP21[up2Size++] = dUtils::AigNodeLitCond(B1[j], 0); }
                    BreakWhen(w, 0, nWords, ( pu0[w]&~pu1[w]) & ~puRoot[w]);
                    if(w==nWords){UP20[up2Size] = dUtils::AigNodeLitCond(B1[i], 0); UP21[up2Size++] = dUtils::AigNodeLitCond(B1[j], 1); }
                    BreakWhen(w, 0, nWords, ( pu0[w]| pu1[w]) & ~puRoot[w]);
                    if(w==nWords){UP20[up2Size] = dUtils::AigNodeLitCond(B1[i], 1); UP21[up2Size++] = dUtils::AigNodeLitCond(B1[j], 1); }
                }

               assert(up2Size<DIV2_MAX);
            }

        }
        queueSize[idx*5+3] = up2Size;

        // check the pos divisor with or gate
        for(int i=0; i<up1Size; i++){
            // the truth table of pu0 has been negated if need, so don't need to consider its polarity
            unsigned *pu0 = divTruthLocal+ dUtils::AigNodeID(UP1[i]) *nWords;
            for(int j=0; j<up2Size; j++){
                unsigned* pu1 = divTruthLocal+dUtils::AigNodeID(UP20[j])*nWords;
                unsigned* pu2 = divTruthLocal+dUtils::AigNodeID(UP21[j])*nWords;
                if(dUtils::AigNodeIsComplement(UP20[j]) && dUtils::AigNodeIsComplement(UP21[j])){
                    BreakWhen(w, 0, nWords, ( pu0[w] | ( pu1[w] |  pu2[w])) ^ puRoot[w]);}
                else if(dUtils::AigNodeIsComplement(UP20[j])){
                    BreakWhen(w, 0, nWords, ( pu0[w] | (~pu1[w] &  pu2[w])) ^ puRoot[w]);}
                else if(dUtils::AigNodeIsComplement(UP21[j])){
                    BreakWhen(w, 0, nWords, ( pu0[w] | ( pu1[w] & ~pu2[w])) ^ puRoot[w]);}
                else{
                    BreakWhen(w, 0, nWords, ( pu0[w] | ( pu1[w] &  pu2[w])) ^ puRoot[w]);}
                if(w==nWords){
                    int gain = MFFCSize - 2;
                    vGain[rootId] = gain;
                    vUsedNodes[rootId] = 6;
                    buildDecGraph22(decGraph + rootId*DECGRAPH_SIZE,
                        dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]),
                        dUtils::AigNodeLitCond(divisorLocal[UP20[j]>>1], nodePhaseLocal[UP20[j]>>1]^dUtils::AigNodeIsComplement(UP20[j])),
                        dUtils::AigNodeLitCond(divisorLocal[UP21[j]>>1], nodePhaseLocal[UP21[j]>>1]^dUtils::AigNodeIsComplement(UP21[j])),
                        dUtils::AigNodeLitCond(divisorLocal[UP1[i]>>1], nodePhaseLocal[UP1[i]>>1]^dUtils::AigNodeIsComplement(UP1[i])),
                        1
                    );
                    terminate = true;
                    vMaskValid[ divisorLocal[UP1[i]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UP20[j]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UP21[j]>>1] ] = STATUS_SKIP;
                    if(verbose>=3)    printf("find 2-resub %d <- (%s%d and %s%d) or %s%d, gain: %d\n", rootId, 
                                            dUtils::AigNodeIsComplement(UP20[j])?"!":" ", divisorLocal[UP20[j]>>1], 
                                            dUtils::AigNodeIsComplement(UP21[j])?"!":" " ,divisorLocal[UP21[j]>>1],
                                            dUtils::AigNodeIsComplement(UP1[i])?"!":" " ,divisorLocal[UP1[i]>>1],
                                            gain);

                }
                if(terminate)   break;
            }
            if(terminate)   break;
        }
        if(terminate)   continue;



        for(int i=0; i<b1Size; i++){
            unsigned *pu0 = divTruthLocal + B1[i] *nWords;
            for(int j=i+1; j<b1Size; j++){
                unsigned *pu1 = divTruthLocal + B1[j]*nWords;

                if(un2Size<DIV2_MAX-4){
                    BreakWhen(w, 0, nWords, ~( pu0[w]| pu1[w]) & puRoot[w]);
                    if(w==nWords){UN20[un2Size] = dUtils::AigNodeLitCond(B1[i], 0); UN21[un2Size++] = dUtils::AigNodeLitCond(B1[j], 0); }
                    BreakWhen(w, 0, nWords, ~(~pu0[w]| pu1[w]) & puRoot[w]);
                    if(w==nWords){UN20[un2Size] = dUtils::AigNodeLitCond(B1[i], 1); UN21[un2Size++] = dUtils::AigNodeLitCond(B1[j], 0); }
                    BreakWhen(w, 0, nWords, ~( pu0[w]|~pu1[w]) & puRoot[w]);
                    if(w==nWords){UN20[un2Size] = dUtils::AigNodeLitCond(B1[i], 0); UN21[un2Size++] = dUtils::AigNodeLitCond(B1[j], 1); }
                    BreakWhen(w, 0, nWords, ~(~pu0[w]|~pu1[w]) & puRoot[w]);
                    if(w==nWords){UN20[un2Size] = dUtils::AigNodeLitCond(B1[i], 1); UN21[un2Size++] = dUtils::AigNodeLitCond(B1[j], 1); }
                }
            //    assert(up2Size<DIV2_MAX);
                assert(un2Size<DIV2_MAX);
            }
        }
        // queueSize[idx*5+3] = up2Size;
        queueSize[idx*5+4] = un2Size;

        // check the neg divisor with and gate
        for(int i=0; i<un1Size; i++){
            unsigned *pu0 = divTruthLocal+ dUtils::AigNodeID(UN1[i]) *nWords;
            for(int j=0; j<un2Size; j++){
                unsigned* pu1 = divTruthLocal+dUtils::AigNodeID(UN20[j])*nWords;
                unsigned* pu2 = divTruthLocal+dUtils::AigNodeID(UN21[j])*nWords;
                if(dUtils::AigNodeIsComplement(UN20[j]) && dUtils::AigNodeIsComplement(UN21[j])){
                    BreakWhen(w, 0, nWords, ( pu0[w] & (~pu1[w] | ~pu2[w])) ^ puRoot[w]);}
                else if(dUtils::AigNodeIsComplement(UN20[j])){
                    BreakWhen(w, 0, nWords, ( pu0[w] & (~pu1[w] |  pu2[w])) ^ puRoot[w]);}
                else if(dUtils::AigNodeIsComplement(UN21[j])){
                    BreakWhen(w, 0, nWords, ( pu0[w] & ( pu1[w] | ~pu2[w])) ^ puRoot[w]);}
                else{
                    BreakWhen(w, 0, nWords, ( pu0[w] & ( pu1[w] |  pu2[w])) ^ puRoot[w]);}
                if(w==nWords){
                    int gain = MFFCSize - 2;
                    vGain[rootId] = gain;
                    vUsedNodes[rootId] = 7;
                    buildDecGraph22(decGraph + rootId*DECGRAPH_SIZE,
                        dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]),
                        dUtils::AigNodeLitCond(divisorLocal[UN20[j]>>1], nodePhaseLocal[UN20[j]>>1]^dUtils::AigNodeIsComplement(UN20[j])),
                        dUtils::AigNodeLitCond(divisorLocal[UN21[j]>>1], nodePhaseLocal[UN21[j]>>1]^dUtils::AigNodeIsComplement(UN21[j])),
                        dUtils::AigNodeLitCond(divisorLocal[UN1[i]>>1], nodePhaseLocal[UN1[i]>>1]^dUtils::AigNodeIsComplement(UN1[i])),
                        0
                    );
                    terminate = true;
                    vMaskValid[ divisorLocal[UN1[i]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UN20[j]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UN21[j]>>1] ] = STATUS_SKIP;
                    if(verbose>=3)    printf("find 2-resub %d <- (%s%d or %s%d) and %s%d, gain: %d\n", rootId, 
                                            dUtils::AigNodeIsComplement(UN20[j])?"!":" ", divisorLocal[UN20[j]>>1], 
                                            dUtils::AigNodeIsComplement(UN21[j])?"!":" " ,divisorLocal[UN21[j]>>1],
                                            dUtils::AigNodeIsComplement(UN1[i])?"!":" " ,divisorLocal[UN1[i]>>1],
                                            gain);

                }
                if(terminate)   break;
            }
            if(terminate)   break;
        }

 
    }                            

}





__global__ void resubDivs3(unsigned* vTruth, const int* vValidEnumInd, const int nValid, int* vCutSize,
                            int* divisor, unsigned* divisorTruth, int* divisorSize, int* simSize, int nMaxCutSize,
                            int* vMaskValid, const int* vMFFCNumSaved,
                            int* decGraph, int* nodePhase, int* vGain, int verbose,
                            int* UP20Global, int* UP21Global, int* UN20Global, int* UN21Global,  
                            int* queueSize, int* vUsedNodes){

    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId;
    int * divisorLocal;
    int w, nWords, nWordsElem = dUtils::TruthWordNum(nMaxCutSize);
    unsigned *divTruthLocal;
    unsigned* puRoot;
    int* nodePhaseLocal;

    // store the idx of divisor; attention it has polarity
    bool terminate;
    int *UP20, *UP21, *UN20, *UN21;
    int up2Size=0, un2Size=0;
    int flag;
 
    for (; idx < nValid; idx += nThreads) {
        rootId = vValidEnumInd[idx];
        UP20 = UP20Global + idx*DIV2_MAX;   UP21 = UP21Global + idx*DIV2_MAX;
        UN20 = UN20Global + idx*DIV2_MAX;   UN21 = UN21Global + idx*DIV2_MAX;
        assert(vMaskValid[rootId]==STATUS_VALID || vMaskValid[rootId]==STATUS_SKIP);
        if(vGain[rootId]) continue;   // skip the node has found a resub
        int rootIdx = simSize[idx]-1;
        int MFFCSize = vMFFCNumSaved[rootId];
        if(MFFCSize<=3)  continue;
        divisorLocal = divisor + idx * DIVISOR_SIZE;
        divTruthLocal = divisorTruth + (long unsigned int)idx*DIVISOR_SIZE*nWordsElem;
        int nVars = vCutSize[idx];
        nodePhaseLocal = nodePhase + idx*DIVISOR_SIZE;
        terminate = false;
        nWords = dUtils::TruthWordNum(nVars);
        puRoot = vTruth+idx*nWordsElem;
        up2Size = queueSize[idx*5+3];
        un2Size = queueSize[idx*5+4];
        flag = 0;

        for(int i=0; i<up2Size; i++){
            unsigned* pu0 = divTruthLocal+dUtils::AigNodeID(UP20[i])*nWords;
            unsigned* pu1 = divTruthLocal+dUtils::AigNodeID(UP21[i])*nWords;
            flag = ( dUtils::AigNodeIsComplement(UP20[i]) << 3 ) | ( dUtils::AigNodeIsComplement(UP21[i]) << 2);

            for(int j=i+1; j<up2Size; j++){
                unsigned* pu2 = divTruthLocal+dUtils::AigNodeID(UP20[j])*nWords;
                unsigned* pu3 = divTruthLocal+dUtils::AigNodeID(UP21[j])*nWords;
                flag = (flag & 12) | ( dUtils::AigNodeIsComplement(UP20[j]) << 1 ) | ( dUtils::AigNodeIsComplement(UP21[j]) << 0);
                switch (flag)
                {
                case 0: // 0000
                    BreakWhen(w, 0, nWords, ( (pu0[w] & pu1[w]) | ( pu2[w] & pu3[w]) ) ^ puRoot[w]);
                    break;
                case 1: // 0001
                    BreakWhen(w, 0, nWords, ( (pu0[w] & pu1[w]) | ( pu2[w] &~pu3[w]) ) ^ puRoot[w]);
                    break;
                case 2: // 0010
                    BreakWhen(w, 0, nWords, ( (pu0[w] & pu1[w]) | (~pu2[w] & pu3[w]) ) ^ puRoot[w]);
                    break;
                case 3: // 0011
                    BreakWhen(w, 0, nWords, ( (pu0[w] & pu1[w]) | ( pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;

                case 4: // 0100
                    BreakWhen(w, 0, nWords, ( (pu0[w] &~pu1[w]) | ( pu2[w] & pu3[w]) ) ^ puRoot[w]);
                    break;
                case 5: // 0101
                    BreakWhen(w, 0, nWords, ( (pu0[w] &~pu1[w]) | ( pu2[w] &~pu3[w]) ) ^ puRoot[w]);
                    break;
                case 6: // 0110
                    BreakWhen(w, 0, nWords, ( (pu0[w] &~pu1[w]) | (~pu2[w] & pu3[w]) ) ^ puRoot[w]);
                    break;
                case 7: // 0111
                    BreakWhen(w, 0, nWords, ( (pu0[w] &~pu1[w]) | ( pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;
                 
                case 8:  // 1000
                    BreakWhen(w, 0, nWords, ( (~pu0[w] & pu1[w]) | ( pu2[w] & pu3[w]) ) ^ puRoot[w]);
                    break;
                case 9:  // 1001
                    BreakWhen(w, 0, nWords, ( (~pu0[w] & pu1[w]) | ( pu2[w] &~pu3[w]) ) ^ puRoot[w]);
                    break;
                case 10: // 1010
                    BreakWhen(w, 0, nWords, ( (~pu0[w] & pu1[w]) | (~pu2[w] & pu3[w]) ) ^ puRoot[w]);
                    break;
                case 11: // 1011
                    BreakWhen(w, 0, nWords, ( (~pu0[w] & pu1[w]) | ( pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;

                case 12: // 1100
                    BreakWhen(w, 0, nWords, ( (pu0[w] | pu1[w]) | ( pu2[w] & pu3[w]) ) ^ puRoot[w]);
                    break;
                case 13: // 1101
                    BreakWhen(w, 0, nWords, ( (pu0[w] | pu1[w]) | ( pu2[w] &~pu3[w]) ) ^ puRoot[w]);
                    break;
                case 14: // 1110
                    BreakWhen(w, 0, nWords, ( (pu0[w] | pu1[w]) | (~pu2[w] & pu3[w]) ) ^ puRoot[w]);
                    break;
                case 15: // 1111
                    BreakWhen(w, 0, nWords, ( (pu0[w] | pu1[w]) | ( pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;

                default:
                    assert(false);
                    break;
                }
                if(w==nWords){
                    int gain = MFFCSize - 3;
                    vGain[rootId] = gain;
                    vUsedNodes[rootId] = 8;
                    buildDecGraph(decGraph + rootId*DECGRAPH_SIZE,
                        dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]),
                        dUtils::AigNodeLitCond(divisorLocal[UP20[i]>>1], nodePhaseLocal[UP20[i]>>1]^dUtils::AigNodeIsComplement(UP20[i])),
                        dUtils::AigNodeLitCond(divisorLocal[UP21[i]>>1], nodePhaseLocal[UP21[i]>>1]^dUtils::AigNodeIsComplement(UP21[i])),
                        dUtils::AigNodeLitCond(divisorLocal[UP20[j]>>1], nodePhaseLocal[UP20[j]>>1]^dUtils::AigNodeIsComplement(UP20[j])),
                        dUtils::AigNodeLitCond(divisorLocal[UP21[j]>>1], nodePhaseLocal[UP21[j]>>1]^dUtils::AigNodeIsComplement(UP21[j])),
                        1
                    );
                    terminate = true;
                    vMaskValid[ divisorLocal[UP20[i]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UP21[i]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UP20[j]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UP21[j]>>1] ] = STATUS_SKIP;
                    if(verbose>=3)    printf("find 3-resub %d <- (%s%d and %s%d) or (%s%d and %s%d), gain: %d\n", rootId, 
                                            dUtils::AigNodeIsComplement(UP20[i])?"!":" " ,divisorLocal[UP20[i]>>1],
                                            dUtils::AigNodeIsComplement(UP21[i])?"!":" " ,divisorLocal[UP21[i]>>1],
                                            dUtils::AigNodeIsComplement(UP20[j])?"!":" ", divisorLocal[UP20[j]>>1], 
                                            dUtils::AigNodeIsComplement(UP21[j])?"!":" " ,divisorLocal[UP21[j]>>1],
                                            gain);
                }
                if(terminate)   break;
            }
            if(terminate)   break;
        }
        if(terminate)   continue;
    
        for(int i=0; i<un2Size; i++){
            unsigned* pu0 = divTruthLocal+dUtils::AigNodeID(UN20[i])*nWords;
            unsigned* pu1 = divTruthLocal+dUtils::AigNodeID(UN21[i])*nWords;
            flag = ( dUtils::AigNodeIsComplement(UN20[i]) << 3 ) | ( dUtils::AigNodeIsComplement(UN21[i]) << 2);

            for(int j=i+1; j<un2Size; j++){
                unsigned* pu2 = divTruthLocal+dUtils::AigNodeID(UN20[j])*nWords;
                unsigned* pu3 = divTruthLocal+dUtils::AigNodeID(UN21[j])*nWords;
                flag = (flag & 12) | ( dUtils::AigNodeIsComplement(UN20[j]) << 1 ) | ( dUtils::AigNodeIsComplement(UN21[j]) << 0);
                switch (flag)
                {
                case 0: // 0000
                    BreakWhen(w, 0, nWords, ( (pu0[w] | pu1[w]) & ( pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;
                case 1: // 0001
                    BreakWhen(w, 0, nWords, ( (pu0[w] | pu1[w]) & ( pu2[w] |~pu3[w]) ) ^ puRoot[w]);
                    break;
                case 2: // 0010
                    BreakWhen(w, 0, nWords, ( (pu0[w] | pu1[w]) & (~pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;
                case 3: // 0011
                    BreakWhen(w, 0, nWords, ( (pu0[w] | pu1[w]) & (~pu2[w] |~pu3[w]) ) ^ puRoot[w]);
                    break;

                case 4: // 0100
                    BreakWhen(w, 0, nWords, ( (pu0[w] |~pu1[w]) & ( pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;                                        
                case 5: // 0101                                   
                    BreakWhen(w, 0, nWords, ( (pu0[w] |~pu1[w]) & ( pu2[w] |~pu3[w]) ) ^ puRoot[w]);
                    break;                                        
                case 6: // 0110                                   
                    BreakWhen(w, 0, nWords, ( (pu0[w] |~pu1[w]) & (~pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;                                        
                case 7: // 0111                                   
                    BreakWhen(w, 0, nWords, ( (pu0[w] |~pu1[w]) & (~pu2[w] |~pu3[w]) ) ^ puRoot[w]);
                    break;
                 
                case 8:  // 1000
                    BreakWhen(w, 0, nWords, ( (~pu0[w] | pu1[w]) & ( pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;                                         
                case 9:  // 1001                                   
                    BreakWhen(w, 0, nWords, ( (~pu0[w] | pu1[w]) & ( pu2[w] |~pu3[w]) ) ^ puRoot[w]);
                    break;                                         
                case 10: // 1010                                   
                    BreakWhen(w, 0, nWords, ( (~pu0[w] | pu1[w]) & (~pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;                                         
                case 11: // 1011                                   
                    BreakWhen(w, 0, nWords, ( (~pu0[w] | pu1[w]) & (~pu2[w] |~pu3[w]) ) ^ puRoot[w]);
                    break;

                case 12: // 1100
                    BreakWhen(w, 0, nWords, ( (~pu0[w] | ~pu1[w]) & ( pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;                                        
                case 13: // 1101                                  
                    BreakWhen(w, 0, nWords, ( (~pu0[w] | ~pu1[w]) & ( pu2[w] |~pu3[w]) ) ^ puRoot[w]);
                    break;                                        
                case 14: // 1110                                  
                    BreakWhen(w, 0, nWords, ( (~pu0[w] | ~pu1[w]) & (~pu2[w] | pu3[w]) ) ^ puRoot[w]);
                    break;                                        
                case 15: // 1111                                  
                    BreakWhen(w, 0, nWords, ( (~pu0[w] | ~pu1[w]) & (~pu2[w] |~pu3[w]) ) ^ puRoot[w]);
                    break;

                default:
                    assert(false);
                    break;
                }
                if(w==nWords){
                    int gain = MFFCSize - 3;
                    vGain[rootId] = gain;
                    vUsedNodes[rootId] = 9;
                    buildDecGraph(decGraph + rootId*DECGRAPH_SIZE,
                        dUtils::AigNodeLitCond(rootId, nodePhaseLocal[rootIdx]),
                        dUtils::AigNodeLitCond(divisorLocal[UN20[i]>>1], nodePhaseLocal[UN20[i]>>1]^dUtils::AigNodeIsComplement(UN20[i])),
                        dUtils::AigNodeLitCond(divisorLocal[UN21[i]>>1], nodePhaseLocal[UN21[i]>>1]^dUtils::AigNodeIsComplement(UN21[i])),
                        dUtils::AigNodeLitCond(divisorLocal[UN20[j]>>1], nodePhaseLocal[UN20[j]>>1]^dUtils::AigNodeIsComplement(UN20[j])),
                        dUtils::AigNodeLitCond(divisorLocal[UN21[j]>>1], nodePhaseLocal[UN21[j]>>1]^dUtils::AigNodeIsComplement(UN21[j])),
                        0
                    );
                    terminate = true;
                    vMaskValid[ divisorLocal[UN20[i]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UN21[i]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UN20[j]>>1] ] = STATUS_SKIP;
                    vMaskValid[ divisorLocal[UN21[j]>>1] ] = STATUS_SKIP;
                    if(verbose>=3)    printf("find 3-resub %d <- (%s%d or %s%d) and (%s%d or %s%d), gain: %d\n", rootId, 
                                            dUtils::AigNodeIsComplement(UN20[i])?"!":" " ,divisorLocal[UN20[i]>>1],
                                            dUtils::AigNodeIsComplement(UN21[i])?"!":" " ,divisorLocal[UN21[i]>>1],
                                            dUtils::AigNodeIsComplement(UN20[j])?"!":" ", divisorLocal[UN20[j]>>1], 
                                            dUtils::AigNodeIsComplement(UN21[j])?"!":" " ,divisorLocal[UN21[j]>>1],
                                            gain);
                }
                if(terminate)   break;
            }
            if(terminate)   break;
        }
    
    }                            

}


