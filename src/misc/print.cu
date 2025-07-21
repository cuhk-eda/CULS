#include <stdio.h>
#include "print.cuh"

__global__ void printAIGA(const int * pFanin0, const int * pFanin1, 
                         const int * pOuts, int nPIs, int nPOs, int nObjs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0)
        return;
    
    printf("id\tfanin0\tfanin1\n");
    
    for (int i = nPIs + 1; i < nObjs; i++) {
        int lit1 = pFanin0[i];
        int lit2 = pFanin1[i];

        printf("%d\t", i);
        printf("%s%d\t", (lit1 & 1) ? "!" : "", lit1 >> 1);
        printf("%s%d\n", (lit2 & 1) ? "!" : "", lit2 >> 1);
    }
    for (int i = 0; i < nPOs; i++) {
        int lit = pOuts[i];
        int id = lit >> 1;
        printf("%s%d\n", (lit & 1) ? "!" : "", id);
    }

}

__global__ void printAIG(const int * vFanin0, const int * vFanin1, const int * vPOs,
                         const int nNodes, const int nPIs, const int nPOs) {
    printf("-------AIG-------\n");
    printf("id\tfanin0\tfanin1\n");
    for (int i = 0; i <= nPIs; i++) {
        printf("%d\n", i);
    }
    for (int i = nPIs + 1; i < nNodes + nPIs + 1; i++) {
        int lit1 = vFanin0[i];
        int lit2 = vFanin1[i];

        printf("%d\t", i);
        printf("%s%d\t", (lit1 & 1) ? "!" : "", lit1 >> 1);
        printf("%s%d\n", (lit2 & 1) ? "!" : "", lit2 >> 1);
    }
    printf("---POs---\n");
    for (int i = 0; i < nPOs; i++) {
        int lit = vPOs[i];
        int id = lit >> 1;
        printf("%s%d\n", (lit & 1) ? "!" : "", id);
    }
    printf("#nodes = %d\n", nNodes);
    printf("-----------------\n");
}


__global__ void printMffc(int * vCutTable, int * vCutSizes, int * vConeSizes,
                          const int * pFanin0, const int * pFanin1, 
                          int nNodes, int nPIs, int nPOs) {
    int smallConeCount = 0, largeCount = 0;
    for (int i = 0; i < nNodes; i++) {
        int id = i + nPIs + 1;
        // printf("node: %d, saved size: %d | ", id, vConeSizes[id]);
        // for (int j = 0; j < vCutSizes[id]; j++) {
        //     printf("%d ", vCutTable[id * CUT_TABLE_SIZE + j]);
        // }
        // printf("\n");

        if (vConeSizes[id] < 2 && vCutSizes[id] != -1)
            smallConeCount++;
        if (vCutSizes[id] == -1)
            largeCount++;
    }
    printf("Too small cone: %d, too large cut: %d\n", smallConeCount, largeCount);
}
