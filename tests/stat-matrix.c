#include "matrix.h"
#include "stopwatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TEST_ROW 24
#define TEST_COL 24
#define STAT_FROM 8
#define STAT_TO 104
#define SAMPLES_NUM 30

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
    &SSEMatrixProvider,
    &prefetchSSEMatrixProvider,
    &AVXMatrixProvider,
    &prefetchAVXMatrixProvider,
};

int *random_matrix(int row, int col)
{
    int *src = (int *) malloc(row * col * sizeof(int));
    srand(time(NULL));
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            *(src + i * col + j) = rand();
    return src;
}

int main()
{
    /* Verify other implementation correction first */
    MatrixAlgo *verify = matrix_providers[0];
    Matrix *dst, *m, *n, *fixed;

    m = verify->create(TEST_ROW, TEST_COL);
    verify->assign(m, random_matrix(TEST_ROW, TEST_COL));
    n = verify->create(TEST_ROW, TEST_COL);
    verify->assign(n, random_matrix(TEST_ROW, TEST_COL));
    verify->mul(&fixed, m, n);

    for (int i = 1; i < 5; i++) {
        MatrixAlgo *algo = matrix_providers[i];
        algo->mul(&dst, m, n);
        if (!algo->equal(dst, fixed)) {
            printf("matrix_providers[%d] Verification failed\n", i);
            return -1;
        }
    }
    printf("All implementation Verification Success\n");

    /* Stat performance */
    double _stat[(STAT_TO - STAT_FROM) / 4 + 1][5];
    int count = (STAT_TO - STAT_FROM) / 4 + 1;
    watch_p ctx = Stopwatch.create();

    for (int i = 0; i < count; i++) {
        int matrix_size = 8 + i * 4;
        int *_a = random_matrix(matrix_size, matrix_size);
        int *_b = random_matrix(matrix_size, matrix_size);
        for (int j = 0; j < 5; j++) {
            MatrixAlgo *algo = matrix_providers[j];
            Matrix *l = algo->create(matrix_size, matrix_size);
            Matrix *r = algo->create(matrix_size, matrix_size);
            algo->assign(l, _a);
            algo->assign(r, _b);
            double tmp = 0;
            for (int k = 0; k < SAMPLES_NUM; k++) {
                Stopwatch.start(ctx);
                algo->mul(&dst, l, r);
                tmp += Stopwatch.read(ctx);
                Stopwatch.reset(ctx);
            }
            _stat[i][j] = tmp / SAMPLES_NUM;
        }
    }
    Stopwatch.destroy(ctx);

    /* Output */
    FILE *fout = fopen("data.csv", "w+");
    for (int i = 0; i < count; i++) {
        int matrix_size = 8 + i * 4;
        fprintf(fout, "%d %.2f %.2f %.2f %.2f %.2f\n",
                matrix_size,
                _stat[i][0], _stat[i][1], _stat[i][2],
                _stat[i][3], _stat[i][4]);
    }
    fclose(fout);

    return 0;
}
