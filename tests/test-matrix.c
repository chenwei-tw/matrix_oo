#include "matrix.h"
#include <stdio.h>

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
};

int main()
{
    MatrixAlgo *algo = matrix_providers[0];

    Matrix *dst, *m, *n, *fixed;
    float a[4][3] = {{1, 2, 3},
        {5, 6, 7},
        {1, 2, 3},
        {5, 6, 7}
    };
    float b[3][4] = {{1, 2, 3, 4},
        {5, 6, 7, 8},
        {1, 2, 3, 4}
    };
    float c[4][4] = {{14, 20, 26, 32},
        {42, 60, 78, 96},
        {14, 20, 26, 32},
        {42, 60, 78, 96}
    };

    /* m: 4x3 */
    //m = algo->create(4, 3);
    algo->assign(&m, (float **) a, 4, 3);

    /* n: 3x4 */
    //n = algo->create(3, 4);
    algo->assign(&n, (float **) b, 3, 4);

    dst = algo->mul(m, n);

    //fixed = algo->create(4, 4);
    algo->assign(&fixed, (float **) c, 4, 4);

    if (algo->equal(dst, fixed))
        return 0;
    return -1;
}
