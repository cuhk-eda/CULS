#pragma once
#include "sop.cuh"
#include "misc/vectors.cuh"
#include "misc/truth_utils.cuh"

namespace sop {

const int SUBG_CAP = MAX_SUBG_SIZE;

// *************** Algebraic Factoring algorithm ***************

__device__ __forceinline__ 
int sopFactorRec(Sop * cSop, int nLits, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem, 
                 subgUtil::Subg<SUBG_CAP> * subg);
__device__ __forceinline__ 
int sopFactorLFRec(Sop * cSop, Sop * cSimple, int nLits, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem, 
                   subgUtil::Subg<SUBG_CAP> * subg);

__device__ __forceinline__ 
void sopCreateInverse(Sop * cResult, unsigned * vInput, int nInputCubes, 
                      VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    unsigned uCube, uMask = 0;
    // start the cover
    cResult->nCubes = 0;
    cResult->pCubes = vecsMem->fetch(nInputCubes);

    for (int i = 0; i < nInputCubes; i++){
        uCube = vInput[i];
        uMask = ((uCube | (uCube >> 1)) & 0x55555555);
        uMask |= (uMask << 1);
        cResult->pCubes[cResult->nCubes++] = uCube ^ uMask;
    }
}











__device__ __forceinline__ unsigned sopCommonCube(Sop * cSop) {
    unsigned uMask;
    int i;
    uMask = ~(unsigned)0;
    // Kit_SopForEachCube( cSop, uCube, i )
    for (i = 0; i < cSop->nCubes; i++)
        uMask &= cSop->pCubes[i];
    return uMask;
}

__device__ __forceinline__ void sopMakeCubeFree(Sop * cSop) {
    unsigned uMask;
    int i;
    uMask = sopCommonCube(cSop);
    if ( uMask == 0 )
        return;
    // remove the common cube
    // Kit_SopForEachCube( cSop, uCube, i )
    for (i = 0; i < cSop->nCubes; i++)
        cSop->pCubes[i] = subgUtil::cubeSharp(cSop->pCubes[i], uMask);
        // Kit_SopWriteCube( cSop, Kit_CubeSharp(uCube, uMask), i );
}

__device__ __forceinline__ void sopDivideByLiteralQuo(Sop * cSop, int iLit) {
    int i, k = 0;
    // Kit_SopForEachCube( cSop, uCube, i )
    for (i = 0; i < cSop->nCubes; i++)
        if (subgUtil::cubeHasLit(cSop->pCubes[i], iLit))
            cSop->pCubes[k++] = subgUtil::cubeRemLit(cSop->pCubes[i], iLit);
    cSop->nCubes = k;
}

__device__ __forceinline__ int sopWorstLiteral(Sop * cSop, int nLits) {
    int i, k, iMin, nLitsMin, nLitsCur;
    int fUseFirst = 1;

    // go through each literal
    iMin = -1;
    nLitsMin = 1000000;
    for (i = 0; i < nLits; i++) {
        // go through all the cubes
        nLitsCur = 0;
        // Kit_SopForEachCube( cSop, uCube, k )
        for (k = 0; k < cSop->nCubes; k++)
            if (subgUtil::cubeHasLit(cSop->pCubes[k], i))
                nLitsCur++;
        // skip the literal that does not occur or occurs once
        if (nLitsCur < 2)
            continue;
        // check if this is the best literal
        if (fUseFirst) {
            if (nLitsMin > nLitsCur) {
                nLitsMin = nLitsCur;
                iMin = i;
            }
        }
        else {
            if (nLitsMin >= nLitsCur) {
                nLitsMin = nLitsCur;
                iMin = i;
            }
        }
    }
    if (nLitsMin < 1000000)
        return iMin;
    return -1;
}

__device__ __forceinline__ void sopDivisorZeroKernelRec(Sop * cSop, int nLits) {
    int iLit;
    // find any literal that occurs at least two times
    iLit = sopWorstLiteral(cSop, nLits);
    if ( iLit == -1 )
        return;
    // derive the cube-free quotient
    sopDivideByLiteralQuo(cSop, iLit); // the same cover
    sopMakeCubeFree(cSop);             // the same cover
    // call recursively
    sopDivisorZeroKernelRec(cSop, nLits);    // the same cover
}

__device__ __forceinline__ 
void sopDup(Sop * cResult, Sop * cSop, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    int i;
    // start the cover
    cResult->nCubes = 0;
    cResult->pCubes = vecsMem->fetch(cSop->nCubes);
    // add the cubes
    // Kit_SopForEachCube( cSop, uCube, i )
    for (i = 0; i < cSop->nCubes; i++)
        // Kit_SopPushCube( cResult, uCube );
        cResult->pCubes[cResult->nCubes++] = cSop->pCubes[i];
}

__device__ __forceinline__ int sopAnyLiteral(Sop * cSop, int nLits) {
    int i, k, nLitsCur;
    // go through each literal
    for (i = 0; i < nLits; i++) {
        // go through all the cubes
        nLitsCur = 0;
        for (k = 0; k < cSop->nCubes; k++)
            if (subgUtil::cubeHasLit(cSop->pCubes[k], i))
                nLitsCur++;
        if (nLitsCur > 1)
            return i;
    }
    return -1;
}

__device__ __forceinline__
int sopDivisor(Sop * cResult, Sop * cSop, int nLits, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    if (cSop->nCubes <= 1)
        return 0;
    if (sopAnyLiteral(cSop, nLits) == -1)
        return 0;
    // duplicate the cover
    sopDup(cResult, cSop, vecsMem);
    // perform the kerneling
    sopDivisorZeroKernelRec(cResult, nLits);
    assert(cResult->nCubes > 0);
    return 1;
}



__device__ __forceinline__
int sopFactorTrivialCubeRec(unsigned uCube, int nStart, int nFinish, subgUtil::Subg<SUBG_CAP> * subg) {
    // printf("enter trivial cube, start=%d, end=%d\n", nStart, nFinish);
    int eNode1, eNode2;
    int i, iLit = -1, nLits, nLits1;
    assert(uCube);
    // count the number of literals in this interval
    nLits = 0;
    for (i = nStart; i < nFinish; i++)
        if (subgUtil::cubeHasLit(uCube, i)) {
            iLit = i;
            nLits++;
        }
    assert(iLit != -1);
    // quit if there is only one literal        
    if (nLits == 1) {
        // printf("return iLit = %d\n", iLit);
        return iLit;
    }
        // return Kit_EdgeCreate( iLit/2, iLit%2 ); // CST
    // split the literals into two parts
    nLits1 = nLits/2;
    // nLits2 = nLits - nLits1;
    // find the splitting point
    nLits = 0;
    for (i = nStart; i < nFinish; i++)
        if (subgUtil::cubeHasLit(uCube, i)) {
            if (nLits == nLits1)
                break;
            nLits++;
        }
    // recursively construct the tree for the parts
    eNode1 = sopFactorTrivialCubeRec(uCube, nStart, i, subg);
    eNode2 = sopFactorTrivialCubeRec(uCube, i, nFinish, subg);
    return subg->addNodeAnd(eNode1, eNode2);
}

__device__ __forceinline__
int sopFactorTrivialRec(unsigned * pCubes, int nCubes, int nLits, subgUtil::Subg<SUBG_CAP> * subg) {
    // printf("enter trivial\n");
    int eNode1, eNode2;
    int nCubes1, nCubes2;
    if (nCubes == 1)
        return sopFactorTrivialCubeRec(pCubes[0], 0, nLits, subg);
    // split the cubes into two parts
    nCubes1 = nCubes/2;
    nCubes2 = nCubes - nCubes1;

    // recursively construct the tree for the parts
    eNode1 = sopFactorTrivialRec(pCubes,           nCubes1, nLits, subg);
    eNode2 = sopFactorTrivialRec(pCubes + nCubes1, nCubes2, nLits, subg);
    return subg->addNodeOr(eNode1, eNode2);
}


__device__ __forceinline__
void sopDivideByCube(Sop * cSop, Sop * cDiv, Sop * vQuo, Sop * vRem, 
                     VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    unsigned uCube, uDiv;
    int i;
    // get the only cube
    assert(cDiv->nCubes == 1);
    uDiv = cDiv->pCubes[0];
    // allocate covers
    vQuo->nCubes = 0;
    vQuo->pCubes = vecsMem->fetch(cSop->nCubes);
    vRem->nCubes = 0;
    vRem->pCubes = vecsMem->fetch(cSop->nCubes);
    // Kit_SopForEachCube( cSop, uCube, i )
    for (i = 0; i < cSop->nCubes; i++) {
        uCube = cSop->pCubes[i];
        if (subgUtil::cubeContains(uCube, uDiv))
            // Kit_SopPushCube( vQuo, Kit_CubeSharp(uCube, uDiv) );
            vQuo->pCubes[vQuo->nCubes++] = subgUtil::cubeSharp(uCube, uDiv);
        else
            // Kit_SopPushCube( vRem, uCube );
            vRem->pCubes[vRem->nCubes++] = uCube;
    }
}

__device__ __forceinline__
void sopDivideInternal(Sop * cSop, Sop * cDiv, Sop * vQuo, Sop * vRem, 
                       VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    unsigned uCube, uDiv;
    unsigned uCube2 = 0; // Suppress "might be used uninitialized"
    unsigned uDiv2, uQuo;
    int i, i2, k, k2, nCubesRem;
    assert(cSop->nCubes >= cDiv->nCubes);
    // consider special case
    if (cDiv->nCubes == 1) {
        sopDivideByCube(cSop, cDiv, vQuo, vRem, vecsMem);
        return;
    }
    // allocate quotient
    vQuo->nCubes = 0;
    vQuo->pCubes = vecsMem->fetch(cSop->nCubes / cDiv->nCubes);
    // for each cube of the cover
    // it either belongs to the quotient or to the remainder
    // Kit_SopForEachCube( cSop, uCube, i )
    for (i = 0; i < cSop->nCubes; i++) {
        uCube = cSop->pCubes[i];
        // skip taken cubes
        if (subgUtil::cubeIsMarked(uCube))
            continue;
        // find a matching cube in the divisor
        uDiv = ~0;
        // Kit_SopForEachCube( cDiv, uDiv, k )
        for (k = 0; k < cDiv->nCubes; k++) {
            uDiv = cDiv->pCubes[k];
            if (subgUtil::cubeContains(uCube, uDiv))
                break;
        }
        // the cube is not found 
        if (k == cDiv->nCubes)
            continue;
        // the quotient cube exists
        uQuo = subgUtil::cubeSharp(uCube, uDiv);
        // find corresponding cubes for other cubes of the divisor
        uDiv2 = ~0;
        // Kit_SopForEachCube( cDiv, uDiv2, k2 )
        for (k2 = 0; k2 < cDiv->nCubes; k2++) {
            uDiv2 = cDiv->pCubes[k2];
            if (k2 == k) continue;
            // find a matching cube
            // Kit_SopForEachCube( cSop, uCube2, i2 )
            for (i2 = 0; i2 < cSop->nCubes; i2++) {
                uCube2 = cSop->pCubes[i2];
                // skip taken cubes
                if (subgUtil::cubeIsMarked(uCube2))
                    continue;
                // check if the cube can be used
                if (subgUtil::cubeContains(uCube2, uDiv2) && uQuo == subgUtil::cubeSharp(uCube2, uDiv2))
                    break;
            }
            // the case when the cube is not found
            if (i2 == cSop->nCubes)
                break;
        }
        // we did not find some cubes - continue looking at other cubes
        if (k2 != cDiv->nCubes)
            continue;
        // we found all cubes - add the quotient cube
        // Kit_SopPushCube( vQuo, uQuo );
        vQuo->pCubes[vQuo->nCubes++] = uQuo;

        // mark the first cube
        // Kit_SopWriteCube( cSop, Kit_CubeMark(uCube), i );
        cSop->pCubes[i] = subgUtil::cubeMark(uCube);
        // mark other cubes that have this quotient
        // Kit_SopForEachCube( cDiv, uDiv2, k2 )
        for (k2 = 0; k2 < cDiv->nCubes; k2++) {
            uDiv2 = cDiv->pCubes[k2];
            if (k2 == k) continue;
            // find a matching cube
            // Kit_SopForEachCube( cSop, uCube2, i2 )
            for (i2 = 0; i2 < cSop->nCubes; i2++) {
                uCube2 = cSop->pCubes[i2];
                // skip taken cubes
                if (subgUtil::cubeIsMarked(uCube2))
                    continue;
                // check if the cube can be used
                if (subgUtil::cubeContains(uCube2, uDiv2) && uQuo == subgUtil::cubeSharp(uCube2, uDiv2))
                    break;
            }
            assert(i2 < cSop->nCubes);
            // the cube is found, mark it 
            // (later we will add all unmarked cubes to the remainder)
            // Kit_SopWriteCube( cSop, Kit_CubeMark(uCube2), i2 );
            cSop->pCubes[i2] = subgUtil::cubeMark(uCube2);
        }
    }
    // determine the number of cubes in the remainder
    nCubesRem = cSop->nCubes - vQuo->nCubes * cDiv->nCubes;
    // allocate remainder
    vRem->nCubes = 0;
    vRem->pCubes = vecsMem->fetch(nCubesRem);
    // finally add the remaining unmarked cubes to the remainder 
    // and clean the marked cubes in the cover
    // Kit_SopForEachCube( cSop, uCube, i )
    for (i = 0; i < cSop->nCubes; i++) {
        uCube = cSop->pCubes[i];
        if (!subgUtil::cubeIsMarked(uCube)) {
            // Kit_SopPushCube( vRem, uCube );
            vRem->pCubes[vRem->nCubes++] = uCube;
            continue;
        }
        // Kit_SopWriteCube( cSop, Kit_CubeUnmark(uCube), i );
        cSop->pCubes[i] = subgUtil::cubeUnmark(uCube);
    }
    assert(nCubesRem == vRem->nCubes);
}

__device__ __forceinline__ int sopBestLiteral(Sop * cSop, int nLits, unsigned uMask) {
    int i, k, iMax, nLitsMax, nLitsCur;
    int fUseFirst = 1;

    // go through each literal
    iMax = -1;
    nLitsMax = -1;
    for (i = 0; i < nLits; i++) {
        if (!subgUtil::cubeHasLit(uMask, i))
            continue;
        // go through all the cubes
        nLitsCur = 0;
        // Kit_SopForEachCube( cSop, uCube, k )
        for (k = 0; k < cSop->nCubes; k++)
            if (subgUtil::cubeHasLit(cSop->pCubes[k], i))
                nLitsCur++;
        // skip the literal that does not occur or occurs once
        if (nLitsCur < 2)
            continue;
        // check if this is the best literal
        if (fUseFirst) {
            if (nLitsMax < nLitsCur) {
                nLitsMax = nLitsCur;
                iMax = i;
            }
        } else {
            if (nLitsMax <= nLitsCur) {
                nLitsMax = nLitsCur;
                iMax = i;
            }
        }
    }
    if (nLitsMax >= 0)
        return iMax;
    return -1;
}

__device__ __forceinline__ 
void sopBestLiteralCover(Sop * cResult, Sop * cSop, unsigned uCube, int nLits, 
                         VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    int iLitBest;
    // get the best literal
    iLitBest = sopBestLiteral(cSop, nLits, uCube);
    // start the cover
    cResult->nCubes = 0;
    cResult->pCubes = vecsMem->fetch(1);
    // set the cube
    // Kit_SopPushCube( cResult, Kit_CubeSetLit(0, iLitBest) );
    cResult->pCubes[cResult->nCubes++] = subgUtil::cubeSetLit(0, iLitBest);
}

__device__ __forceinline__ 
void sopCommonCubeCover(Sop * cResult, Sop * cSop, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem) {
    assert(cSop->nCubes > 0);
    cResult->nCubes = 0;
    cResult->pCubes = vecsMem->fetch(1);
    // Kit_SopPushCube( cResult, Kit_SopCommonCube(cSop) );
    cResult->pCubes[cResult->nCubes++] = sopCommonCube(cSop);
}

__device__ __forceinline__
int sopFactorRec(Sop * cSop, int nLits, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem, 
                 subgUtil::Subg<SUBG_CAP> * subg) {
    Sop Div, Quo, Rem, Com;
    Sop * cDiv = &Div, * cQuo = &Quo, * cRem = &Rem, * cCom = &Com;
    int eNodeDiv, eNodeQuo, eNodeRem, eNodeAnd;

    assert(cSop->nCubes > 0);

    // get the divisor
    if (!sopDivisor(cDiv, cSop, nLits, vecsMem))
        return sopFactorTrivialRec(cSop->pCubes, cSop->nCubes, nLits, subg);
    
    // divide the cover by the divisor
    sopDivideInternal(cSop, cDiv, cQuo, cRem, vecsMem);

    // check the trivial case
    assert(cQuo->nCubes > 0);
    if (cQuo->nCubes == 1)
        return sopFactorLFRec(cSop, cQuo, nLits, vecsMem, subg);
    
    // make the quotient cube ABC_FREE
    sopMakeCubeFree(cQuo);

    // divide the cover by the quotient
    sopDivideInternal(cSop, cQuo, cDiv, cRem, vecsMem);

    // check the trivial case
    // if ( Kit_SopIsCubeFree( cDiv ) )
    if (sopCommonCube(cDiv) == 0) {
        eNodeDiv = sopFactorRec(cDiv, nLits, vecsMem, subg);
        eNodeQuo = sopFactorRec(cQuo, nLits, vecsMem, subg);
        eNodeAnd = subg->addNodeAnd(eNodeDiv, eNodeQuo);
        if (cRem->nCubes == 0)
            return eNodeAnd;
        eNodeRem = sopFactorRec(cRem, nLits, vecsMem, subg);
        return subg->addNodeOr(eNodeAnd, eNodeRem);
    }

    // get the common cube
    sopCommonCubeCover(cCom, cDiv, vecsMem);

    // solve the simple problem
    return sopFactorLFRec(cSop, cCom, nLits, vecsMem, subg);
}

__device__ __forceinline__
int sopFactorLFRec(Sop * cSop, Sop * cSimple, int nLits, VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem, 
                   subgUtil::Subg<SUBG_CAP> * subg) {
    Sop Div, Quo, Rem;
    Sop * cDiv = &Div, * cQuo = &Quo, * cRem = &Rem;
    int eNodeDiv, eNodeQuo, eNodeRem, eNodeAnd;
    assert(cSimple->nCubes == 1);
    // get the most often occurring literal
    sopBestLiteralCover(cDiv, cSop, cSimple->pCubes[0], nLits, vecsMem);
    // divide the cover by the literal
    sopDivideByCube(cSop, cDiv, cQuo, cRem, vecsMem);
    // get the node pointer for the literal
    eNodeDiv = sopFactorTrivialCubeRec(cDiv->pCubes[0], 0, nLits, subg);
    // factor the quotient and remainder
    eNodeQuo = sopFactorRec(cQuo, nLits, vecsMem, subg);
    eNodeAnd = subg->addNodeAnd(eNodeDiv, eNodeQuo);
    if (cRem->nCubes == 0)
        return eNodeAnd;
    eNodeRem = sopFactorRec(cRem, nLits, vecsMem, subg);
    return subg->addNodeOr(eNodeAnd, eNodeRem);
}

__device__ __forceinline__
void sopFactor(unsigned * vCover, int nCoverSize, int fCompl, const int * vCuts, int nVars, 
               VecsMem<unsigned, ISOP_FACTOR_MEM_CAP> * vecsMem, 
               subgUtil::Subg<SUBG_CAP> * subg) {
    // printf("** Start SOP factor ... nVars = %d, cubes = ", nVars);
    // for (int i = 0; i < nCoverSize; i++)
    //     printf("0x%08x ", vCover[i]);
    // printf("\n");

    Sop sop, * cSop = &sop;
    int eRoot;
    assert(nVars < 16);

    // clear subgraph and assign leaves
    subg->nSize = nVars;

    // check for trivial functions
    if (nCoverSize == 0) {
        if (fCompl)
            subg->createConst1();
        else
            subg->createConst0();
        return;
    }
    if (nCoverSize == 1 && vCover[0] == 0) {
        if (fCompl)
            subg->createConst0();
        else
            subg->createConst1();
        return;
    }
    // perform CST
    sopCreateInverse(cSop, vCover, nCoverSize, vecsMem);
    // factor the cover
    eRoot = sopFactorRec(cSop, 2 * nVars, vecsMem, subg);

    // int lit0, lit1, fCompRoot;
    // printf("    subg: ");
    // for (int i = nVars; i < subg->nSize; i++) {
    //     subgUtil::unbindAndNodeKeyFlag(subg->pArray[i], &lit0, &lit1, &fCompRoot);
    //     printf("%s%d,%s%d ", dUtils::AigNodeIsComplement(lit0) ? "!" : "", lit0 >> 1, dUtils::AigNodeIsComplement(lit1) ? "!" : "", lit1 >> 1);
    // }
    // printf("; final node is %scomplemented ", fCompRoot ? "" : "NOT ");
    // printf("; eRoot is %scomplemented\n", dUtils::AigNodeIsComplement(eRoot) ? "" : "NOT ");

    // if eRoot is a leaf, then this is the case of the resyned cut is a const or a single var of cut nodes
    if (dUtils::AigNodeID(eRoot) < nVars) {
        // add one node with lit0 = lit1 = original lit of eRoot to the subgraph
        // note that eRoot will not be const 0/1 due to algebraic factoring
        assert(subg->nSize == nVars);
        subg->createSingleExistingVar(
            dUtils::AigNodeLitCond(vCuts[dUtils::AigNodeID(eRoot)], 
                                   dUtils::AigNodeIsComplement(eRoot) != fCompl)
        );
        return;
    }

    // the complementation info of eRoot is already in the root node
    // if fCompl, do complemnt
    if (fCompl) {
        uint64 rootNode = subg->pArray[subg->nSize - 1];
        int lit0, lit1, fCompRoot;
        subgUtil::unbindAndNodeKeyFlag(rootNode, &lit0, &lit1, &fCompRoot);
        subg->pArray[subg->nSize - 1] = subgUtil::formAndNodeKeyFlag(lit0, lit1, 1 - fCompRoot);
    }
}

} // namespace sop
