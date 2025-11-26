#pragma once
#include "sop.cuh"
#include "misc/vectors.cuh"
#include "misc/truth_utils.cuh"

namespace sop {

// *************** Minato ISOP algorithm ***************

__host__ __device__ inline
unsigned minatoIsop5Rec(unsigned uOn, unsigned uOnDc, int nVars, 
                        Sop * pcRes, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    unsigned uMasks[5] = { 0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000 };
    Sop cRes0, cRes1, cRes2;
    Sop * pcRes0 = &cRes0, * pcRes1 = &cRes1, * pcRes2 = &cRes2;
    unsigned uOn0, uOn1, uOnDc0, uOnDc1, uRes0, uRes1, uRes2;
    int i, k, Var;
    assert(nVars <= 5);
    assert((uOn & ~uOnDc) == 0);

    if (uOn == 0) {
        pcRes->nLits = 0;
        pcRes->nCubes = 0;
        pcRes->pCubes = NULL;
        return 0;
    }
    if (uOnDc == 0xFFFFFFFF) {
        pcRes->nLits = 0;
        pcRes->nCubes = 1;
        pcRes->pCubes = vecsMem->fetch(1);
        VecsMem_OverflowReturnValue(*vecsMem, 0);
        assert(pcRes->pCubes != NULL);

        pcRes->pCubes[0] = 0;
        return 0xFFFFFFFF;
    }

    // find the topmost var
    for (Var = nVars-1; Var >= 0; Var--)
        if (truthUtil::varInSupport(&uOn, 5, Var) || truthUtil::varInSupport(&uOnDc, 5, Var))
             break;
    assert(Var >= 0);

    // cofactor
    uOn0 = uOn1 = uOn;
    uOnDc0 = uOnDc1 = uOnDc;
    truthUtil::cofactor0(&uOn0, Var + 1, Var);
    truthUtil::cofactor1(&uOn1, Var + 1, Var);
    truthUtil::cofactor0(&uOnDc0, Var + 1, Var);
    truthUtil::cofactor1(&uOnDc1, Var + 1, Var);

    // solve for cofactors
    uRes0 = minatoIsop5Rec(uOn0 & ~uOnDc1, uOnDc0, Var, pcRes0, vecsMem);
    VecsMem_OverflowReturnValue(*vecsMem, 0);
    uRes1 = minatoIsop5Rec(uOn1 & ~uOnDc0, uOnDc1, Var, pcRes1, vecsMem);
    VecsMem_OverflowReturnValue(*vecsMem, 0);
    uRes2 = minatoIsop5Rec((uOn0 & ~uRes0) | (uOn1 & ~uRes1), uOnDc0 & uOnDc1, Var, pcRes2, vecsMem);
    VecsMem_OverflowReturnValue(*vecsMem, 0);
    
    // create the resulting cover
    pcRes->nLits  = pcRes0->nLits  + pcRes1->nLits  + pcRes2->nLits + pcRes0->nCubes + pcRes1->nCubes;
    pcRes->nCubes = pcRes0->nCubes + pcRes1->nCubes + pcRes2->nCubes;
    pcRes->pCubes = vecsMem->fetch(pcRes->nCubes);
    VecsMem_OverflowReturnValue(*vecsMem, 0);
    assert(pcRes->pCubes != NULL);

    k = 0;
    for (i = 0; i < pcRes0->nCubes; i++)
        pcRes->pCubes[k++] = pcRes0->pCubes[i] | (1 << ((Var<<1)+0));
    for (i = 0; i < pcRes1->nCubes; i++)
        pcRes->pCubes[k++] = pcRes1->pCubes[i] | (1 << ((Var<<1)+1));
    for (i = 0; i < pcRes2->nCubes; i++)
        pcRes->pCubes[k++] = pcRes2->pCubes[i];
    assert(k == pcRes->nCubes);

    // derive the final truth table
    uRes2 |= (uRes0 & ~uMasks[Var]) | (uRes1 & uMasks[Var]);
    return uRes2;
}

__host__ __device__ inline
unsigned * minatoIsopRec(const unsigned * puOn, const unsigned * puOnDc, int nVars, 
                         Sop * pcRes, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    // puOn = 0, puOnDc = 0, f = 0; puOn = 1, puOnDc = 1, f = 1; puOn = 0, puOnDc = 1, f = DC
    // for details, see Minato's ISOP-BDD paper: 
    //     Fast Generation of Prime-Irredundant Covers from Binary Decision Diagrams

    Sop cRes0, cRes1, cRes2;
    Sop * pcRes0 = &cRes0, * pcRes1 = &cRes1, * pcRes2 = &cRes2;
    unsigned * puRes0, * puRes1, * puRes2;
    const unsigned * puOn0, * puOn1, * puOnDc0, * puOnDc1;
    unsigned * pTemp, * pTemp0, * pTemp1;
    int i, k, Var, nWords, nWordsAll;

    // allocate room for the resulting truth table
    nWordsAll = dUtils::TruthWordNum(nVars);
    pTemp = vecsMem->fetch(nWordsAll);
    VecsMem_OverflowReturnValue(*vecsMem, NULL);
    assert(pTemp != NULL);

    // check for constants
    if (truthUtil::isConst0(puOn, nVars)) {
        pcRes->nLits  = 0;
        pcRes->nCubes = 0;
        pcRes->pCubes = NULL;
        truthUtil::clear(pTemp, nVars);
        return pTemp;
    }
    if (truthUtil::isConst1(puOnDc, nVars)) {
        pcRes->nLits  = 0;
        pcRes->nCubes = 1;
        pcRes->pCubes = vecsMem->fetch(1);
        VecsMem_OverflowReturnValue(*vecsMem, NULL);
        assert(pcRes->pCubes != NULL);

        pcRes->pCubes[0] = 0; // const-true cube contains no literal
        truthUtil::fill(pTemp, nVars);
        return pTemp;
    }

    // find the topmost var
    for (Var = nVars - 1; Var >= 0; Var--)
        if (truthUtil::varInSupport(puOn, nVars, Var) || truthUtil::varInSupport(puOnDc, nVars, Var))
             break;
    assert(Var >= 0);

    // consider a simple case when one-word computation can be used
    if ( Var < 5 ) {
        unsigned uRes = minatoIsop5Rec(puOn[0], puOnDc[0], Var + 1, pcRes, vecsMem);
        VecsMem_OverflowReturnValue(*vecsMem, NULL);
        for (i = 0; i < nWordsAll; i++)
            pTemp[i] = uRes;
        return pTemp;
    }
    assert(Var >= 5);
    nWords = dUtils::TruthWordNum(Var);

    // cofactor
    puOn0   = puOn;    puOn1   = puOn + nWords;
    puOnDc0 = puOnDc;  puOnDc1 = puOnDc + nWords;
    pTemp0  = pTemp;   pTemp1  = pTemp + nWords;

    // solve for cofactors
    truthUtil::truthSharp(pTemp0, puOn0, puOnDc1, Var); // f'0
    puRes0 = minatoIsopRec(pTemp0, puOnDc0, Var, pcRes0, vecsMem); // g0
    VecsMem_OverflowReturnValue(*vecsMem, NULL);
    truthUtil::truthSharp(pTemp1, puOn1, puOnDc0, Var); // f'1
    puRes1 = minatoIsopRec(pTemp1, puOnDc1, Var, pcRes1, vecsMem); // g1
    VecsMem_OverflowReturnValue(*vecsMem, NULL);

    truthUtil::truthSharp(pTemp0, puOn0, puRes0, Var); // f''0
    truthUtil::truthSharp(pTemp1, puOn1, puRes1, Var); // f''1
    truthUtil::truthOr(pTemp0, pTemp0, pTemp1, Var);   // f*
    truthUtil::truthAnd(pTemp1, puOnDc0, puOnDc1, Var);
    puRes2 = minatoIsopRec(pTemp0, pTemp1, Var, pcRes2, vecsMem);
    VecsMem_OverflowReturnValue(*vecsMem, NULL);

    // create the resulting cover
    pcRes->nLits  = pcRes0->nLits  + pcRes1->nLits  + pcRes2->nLits + pcRes0->nCubes + pcRes1->nCubes;
    pcRes->nCubes = pcRes0->nCubes + pcRes1->nCubes + pcRes2->nCubes;
    pcRes->pCubes = vecsMem->fetch(pcRes->nCubes);
    VecsMem_OverflowReturnValue(*vecsMem, NULL);
    assert(pcRes->pCubes != NULL);

    // since there are at most 16 vars, 
    // the 2i+1 (2i) bit indicate whether i-th var('s complement) is in the cube
    k = 0;
    for (i = 0; i < pcRes0->nCubes; i++)
        pcRes->pCubes[k++] = pcRes0->pCubes[i] | (1 << ((Var<<1)+0));
    for (i = 0; i < pcRes1->nCubes; i++)
        pcRes->pCubes[k++] = pcRes1->pCubes[i] | (1 << ((Var<<1)+1));
    for (i = 0; i < pcRes2->nCubes; i++)
        pcRes->pCubes[k++] = pcRes2->pCubes[i];
    assert(k == pcRes->nCubes);

    // create the resulting truth table
    truthUtil::truthOr(pTemp0, puRes0, puRes2, Var);
    truthUtil::truthOr(pTemp1, puRes1, puRes2, Var);
    // copy the table if needed
    nWords <<= 1;
    for (i = 1; i < nWordsAll/nWords; i++)
        for (k = 0; k < nWords; k++)
            pTemp[i*nWords + k] = pTemp[k];
    return pTemp;
}

__host__ __device__ inline
void minatoIsop(const unsigned * puTruth, int nVars, 
                VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    Sop cRes, * pcRes = &cRes;
    unsigned * pResult, * pTemp;

    vecsMem->shrink(0); // clear the memory

    pResult = minatoIsopRec(puTruth, puTruth, nVars, pcRes, vecsMem);
    VecsMem_OverflowReturn(*vecsMem);
    assert(truthUtil::truthEqual(puTruth, pResult, nVars));

    if (pcRes->nCubes == 0 || (pcRes->nCubes == 1 && pcRes->pCubes[0] == 0)) {
        vecsMem->pArray[0] = 0;
        vecsMem->shrink(pcRes->nCubes);
        return;
    }

    // move the cover representation to the beginning of the memory buffer
    pTemp = vecsMem->fetch(pcRes->nCubes);
    VecsMem_OverflowReturn(*vecsMem);
    assert(pTemp != NULL && !vecsMem->overflow());
    for (int i = 0; i < pcRes->nCubes; i++)
        pTemp[i] = pcRes->pCubes[i];
    for (int i = 0; i < pcRes->nCubes; i++)
        vecsMem->pArray[i] = pTemp[i];
    vecsMem->shrink(pcRes->nCubes);
}

} // namespace sop