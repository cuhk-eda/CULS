#pragma once

__global__ void printAIG(const int * vFanin0, const int * vFanin1, const int * vPOs,
                         const int nNodes, const int nPIs, const int nPOs);
__global__ void printAIGA(const int * pFanin0, const int * pFanin1, 
                         const int * pOuts, int nPIs, int nPOs, int nObjs);
__global__ void printMffc(int * vCutTable, int * vCutSizes, int * vConeSizes,
                          const int * pFanin0, const int * pFanin1, 
                          int nNodes, int nPIs, int nPOs);
