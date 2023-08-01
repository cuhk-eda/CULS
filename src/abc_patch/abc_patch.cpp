#include "abc_patch/abc_patch_int.h"

namespace abcPatch {

// gpuls frame
static GpuFrame s_gpuFrame;

AIGMan * getGpuMan() {
    return s_gpuFrame.pMan;
}

bool gpuManIsActive() {
    return s_gpuFrame.active;
}

void gpuManSetActive() {
    s_gpuFrame.active = true;
    s_gpuFrame.pMan->resetTime();
}

void gpuManSetInactive() {
    s_gpuFrame.active = false;
}

void showPatchInfo() {
    printf("Enhanced with GPU-based algorithm implementations, \n");
    printf("by the Chinese University of Hong Kong\n");
}

// called during ABC startup
void patchInitFunc(Abc_Frame_t * pAbc) {
    s_gpuFrame.pMan = new AIGMan();
    s_gpuFrame.active = false;

    registerAllAbcCommands(pAbc);
    showPatchInfo();
}

// called during ABC termination
void patchDestroyFunc(Abc_Frame_t * pAbc) {
    delete s_gpuFrame.pMan;
}

Abc_FrameInitializer_t frameInitializer = {
    patchInitFunc, patchDestroyFunc, NULL, NULL};

// register the initializer a constructor of a global object
// called before main (and ABC startup)
struct AbcRegistrar {
    AbcRegistrar() {
        Abc_FrameAddInitializer(&frameInitializer);
    }
};
static AbcRegistrar s_abcRegistrar;

} // namespace abcPatch
