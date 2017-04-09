#include "matrix.h"
#include <stdio.h>

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
    &SSEMatrixProvider,
};

int main()
{
    MatrixAlgo *algo = matrix_providers[1];

    Matrix *dst, *m, *n, *fixed;

    m = algo->create(4, 3);
    algo->assign(m, (int []) {
        1, 2, 3,
        5, 6, 7,
        1, 2, 3,
        5, 6, 7,
    });

    n = algo->create(3, 4);
    algo->assign(n, (int []) {
        1, 2, 3, 4,
        5, 6, 7, 8,
        1, 2, 3, 4
    });

    algo->mul(&dst, m, n);

    fixed = algo->create(4, 4);
    algo->assign(fixed, (int []) {
        14, 20, 26, 32,
        42, 60, 78, 96,
        14, 20, 26, 32,
        42, 60, 78, 96
    });

    if (algo->equal(dst, fixed))
        return 0;
    return -1;
}
