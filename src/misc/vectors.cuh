#pragma once

template <typename T, int nCap>
struct VecsMem {
    static_assert(nCap <= (1 << 16), "The capacity of VecsMem should be no larger than 2^16!\n");

    __host__ __device__ __forceinline__ T * fetch(int nWords) {
        if (nWords <= 0)
            return NULL;
        if (nSize + nWords > nCap) {
            printf("try to decrease K in refactor !!!\n");
            assert(0);
            return NULL;
        }
        nSize += nWords;
        return pArray + nSize - nWords;
    }

    __host__ __device__ __forceinline__ void shrink(int nSizeNew) {
        assert(nSize >= nSizeNew);
        nSize = nSizeNew;
    }

    int nSize = 0;
    T pArray[nCap];
};
