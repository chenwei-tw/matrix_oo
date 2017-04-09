#include "matrix.h"
#include <stdio.h>

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
};

int main()
{
    MatrixAlgo *algo = matrix_providers[0];

    Matrix *dst, *m, *n, *fixed;

    m = algo->create(4);
    algo->assign(m, (int []) {
        1, 2, 3, 4,
        5, 6, 7, 8,
        1, 2, 3, 4,
        5, 6, 7, 8
    });

    n = algo->create(4);
    algo->assign(n, (int []) {
        1, 2, 3, 4,
        5, 6, 7, 8,
        1, 2, 3, 4,
        5, 6, 7, 8
    });

    dst = algo->create(4);
    algo->mul(dst, m, n);

    fixed = algo->create(4);
    algo->assign(fixed, (int []) {
        34, 44, 54, 64,
        82, 108, 134, 160,
        34, 44, 54, 64,
        82, 108, 134, 160
    });

    if (algo->equal(dst, fixed))
        return 0;
    return -1;
}
