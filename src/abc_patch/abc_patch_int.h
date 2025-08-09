#pragma once
extern "C" {
#include "base/main/main.h"
#include "base/main/mainInt.h"
}

#include "aig_manager.h"

namespace abcPatch {

struct GpuFrame {
    AIGMan * pMan;
    bool active;
};

// declarations
AIGMan * getGpuMan();
bool gpuManIsActive();
void gpuManSetActive();
void gpuManSetInactive();

bool AbcNtkToGpuMan(Abc_Ntk_t * pNtk, AIGMan * pMan);
Abc_Ntk_t * GpuManToAbcNtk(AIGMan * pMan);

void registerAllAbcCommands(Abc_Frame_t * pAbc);

// dedicated helpers
const int AigConst1 = 0;
inline int AigNodeID(int lit) { return lit >> 1; }
inline int AigNodeIsComplement(int lit) { return lit & 1; }
inline int AigNodeLitCond(int nodeId, int complement) {
    return (int)(((unsigned)nodeId << 1) | (unsigned)(complement != 0));
}

} // namespace abcPatch
