#include "abc_patch/abc_patch_int.h"

namespace abcPatch {

bool AbcNtkToGpuMan(Abc_Ntk_t * pNtk, AIGMan * pMan) {
    if (!Abc_NtkIsStrash(pNtk)) {
        printf("Please strash the network before converting to GPU.\n");
        return false;
    }

    int i, id0, id1;
    Abc_Obj_t * pObj, * pObj0, * pObj1;

    pMan->clearDevice();
    pMan->clearHost();
    
    pMan->moduleName = pNtk->pName;
    pMan->modulePath = pNtk->pSpec;
    pMan->nPIs = Abc_NtkCiNum(pNtk);
    pMan->nPOs = Abc_NtkCoNum(pNtk);

    Vec_Ptr_t * vNodes = Abc_NtkDfs(pNtk, 0);
    pMan->nNodes = Vec_PtrSize(vNodes);
    pMan->nObjs = pMan->nPIs + pMan->nNodes + 1;

    // allocate memory
    pMan->allocHost();
    memset(pMan->pFanin0, -1, pMan->nObjs * sizeof(int));
    memset(pMan->pFanin1, -1, pMan->nObjs * sizeof(int));

    // prepare delay data
    std::vector<int> vDelays(pMan->nObjs, -1);
    for (i = 0; i <= pMan->nPIs; i++)
        vDelays[i] = 0;
    
    // process const and PIs
    std::unordered_map<int, int> mAbc2GpuId;
    mAbc2GpuId[Abc_AigConst1(pNtk)->Id] = AigConst1;
    Abc_NtkForEachCi(pNtk, pObj, i) {
        assert(mAbc2GpuId.count(pObj->Id) == 0);
        mAbc2GpuId[pObj->Id] = i + 1;
    }
    
    // create AND nodes
    int maxDelay = -1;
    Vec_PtrForEachEntry(Abc_Obj_t *, vNodes, pObj, i) {
        int nodeId = i + 1 + pMan->nPIs;
        
        pObj0 = Abc_ObjFanin0(pObj);
        assert(mAbc2GpuId.count(pObj0->Id) > 0);
        id0 = mAbc2GpuId[pObj0->Id];
        pMan->pFanin0[nodeId] = AigNodeLitCond(id0, Abc_ObjFaninC0(pObj));

        pObj1 = Abc_ObjFanin1(pObj);
        assert(mAbc2GpuId.count(pObj1->Id) > 0);
        id1 = mAbc2GpuId[pObj1->Id];
        pMan->pFanin1[nodeId] = AigNodeLitCond(id1, Abc_ObjFaninC1(pObj));

        assert(mAbc2GpuId.count(pObj->Id) == 0);
        mAbc2GpuId[pObj->Id] = nodeId;

        ++pMan->pNumFanouts[id0];
        ++pMan->pNumFanouts[id1];

        // propagate delay
        assert(vDelays[id0] != -1 && vDelays[id1] != -1);
        vDelays[nodeId] = 1 + std::max(vDelays[id0], vDelays[id1]);
        maxDelay = std::max(maxDelay, vDelays[nodeId]);
    }
    pMan->nLevels = maxDelay;

    // process POs
    Abc_NtkForEachCo(pNtk, pObj, i) {
        pObj0 = Abc_ObjFanin0(pObj);
        assert(mAbc2GpuId.count(pObj0->Id) > 0);
        id0 = mAbc2GpuId[pObj0->Id];
        pMan->pOuts[i] = AigNodeLitCond(id0, Abc_ObjFaninC0(pObj));
        ++pMan->pNumFanouts[id0];
    }

    Vec_PtrFree(vNodes);
    return true;
}

Abc_Ntk_t * GpuManToAbcNtk(AIGMan * pMan) {
    if (pMan->deviceAllocated) {
        pMan->toHost();
        pMan->clearDevice();
    }

    int i, lit0, lit1;
    Abc_Obj_t * pObj, * pObj0, * pObj1;
    Abc_Ntk_t * pNtk = Abc_NtkAlloc(ABC_NTK_STRASH, ABC_FUNC_AIG, 1);

    pNtk->pName = Extra_UtilStrsav(pMan->moduleName.c_str());
    pNtk->pSpec = Extra_UtilStrsav(pMan->modulePath.c_str());

    // create const, PIs and POs
    std::vector<Abc_Obj_t*> vGpu2AbcObj(pMan->nObjs, NULL);
    std::vector<Abc_Obj_t*> vAbcPos(pMan->nPOs, NULL);

    Abc_AigConst1(pNtk)->Level = 0;
    vGpu2AbcObj[AigConst1] = Abc_AigConst1(pNtk);
    for (i = 1; i <= pMan->nPIs; i++) {
        pObj = Abc_NtkCreatePi(pNtk);
        pObj->Level = 0;
        vGpu2AbcObj[i] = pObj;
    }
    for (i = 0; i < pMan->nPOs; i++) {
        pObj = Abc_NtkCreatePo(pNtk);
        vAbcPos[i] = pObj;
    }

    // create AND nodes
    for (i = pMan->nPIs + 1; i < pMan->nObjs; i++) {
        lit0 = pMan->pFanin0[i];
        lit1 = pMan->pFanin1[i];
        pObj0 = vGpu2AbcObj[AigNodeID(lit0)];
        pObj1 = vGpu2AbcObj[AigNodeID(lit1)];
        assert(pObj0 && pObj1);

        pObj0 = (Abc_Obj_t *)Aig_NotCond(
            (Aig_Obj_t *)pObj0, AigNodeIsComplement(lit0));
        pObj1 = (Abc_Obj_t *)Aig_NotCond(
            (Aig_Obj_t *)pObj1, AigNodeIsComplement(lit1));
        pObj = Abc_AigAnd((Abc_Aig_t *)pNtk->pManFunc, pObj0, pObj1);
        assert(!Aig_IsComplement((Aig_Obj_t *)pObj));

        vGpu2AbcObj[i] = pObj;
        // delay has already been propagated in Abc_AigAnd()
    }
    assert(Abc_NtkCiNum(pNtk) == pMan->nPIs);
    assert(Abc_NtkCoNum(pNtk) == pMan->nPOs);
    assert(Abc_NtkNodeNum(pNtk) <= pMan->nNodes);

    // connect POs
    for (i = 0; i < pMan->nPOs; i++) {
        pObj = vAbcPos[i];
        lit0 = pMan->pOuts[i];
        pObj0 = vGpu2AbcObj[AigNodeID(lit0)];
        assert(pObj0);

        pObj0 = (Abc_Obj_t *)Aig_NotCond(
            (Aig_Obj_t *)pObj0, AigNodeIsComplement(lit0));
        Abc_ObjAddFanin(pObj, pObj0);
    }

    return pNtk;
}

} // namespace abcPatch
