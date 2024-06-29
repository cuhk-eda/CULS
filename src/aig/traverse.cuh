#pragma once
#include <tuple>
#include "common.h"

namespace Aig {

std::tuple<int, int, int *, int *, int *, int *>
levelWiseSchemeCPU(const int * vFanin0, const int * vFanin1, 
                   int nPIs, int nObjs, int * vLevels = NULL);

std::tuple<int, int, int *, int *, int *, int *>
reverseLevelWiseSchemeCPU(const int * vFanin0, const int * vFanin1, const int * vOuts, 
                        int nPIs, int nObjs, int nPOs, int * vLevels=NULL);


void getFanoutGPU(const int* d_pFanin0, const int* d_pFanin1, const int* d_pOuts, const int* d_pNumFanouts, int* vFanoutRanges, int* vFanoutInd, int nPIs, int nObjs, int nPOs, int* vMarks=NULL);

void getFanout(const int* pFanin0, const int* pFanin1, const int* pOuts, const int* d_pNumFanouts, int* vFanoutRanges, int* vFanoutInd, 
                        int nPIs, int nObjs, int nPOs, int nFanoutRange);

} // namespace Aig
