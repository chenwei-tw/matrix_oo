#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

#include <xmmintrin.h>
#include <immintrin.h>

struct avx_priv {
    int **values;
};

#define PRIV(x) \
    ((struct avx_priv *) ((x)->priv))

static Matrix *create(int row, int col)
{
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    int actual_row = (row % 8 == 0) ? row : row + (8 - row % 8);
    int actual_col = (col % 8 == 0) ? col : col + (8 - col % 8);
    m->row = row;
    m->col = col;
    m->priv = malloc(sizeof(struct avx_priv));
    PRIV(m)->values = (int **) malloc(actual_row * sizeof(int *));
    for(int i = 0; i < actual_row; i++)
        PRIV(m)->values[i] = (int *) calloc(actual_col, sizeof(int));
    return m;
}

static void assign(Matrix *thiz, int *data)
{
    for (int i = 0; i < thiz->row; i++)
        for (int j = 0; j < thiz->col; j++)
            PRIV(thiz)->values[i][j] =
                *(data + i * thiz->col + j);
}

static bool equal(const Matrix *l, const Matrix *r)
{
    if (l->col != r->col || l->row != r->row) return false;

    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < l->col; j++)
            if (PRIV(l)->values[i][j] != PRIV(r)->values[i][j])
                return false;
    return true;
}

static void display(Matrix *thiz)
{
    for (int i = 0; i < thiz->row; i++) {
        for (int j = 0; j < thiz->col; j++) {
            printf("%4d ", PRIV(thiz)->values[i][j]);
        }
        printf("\n");
    }
}

static bool avx_mul(Matrix **dst, const Matrix *l, const Matrix *r)
{
    if(l->col != r->row) return false;
    int src1_h = (l->row % 8 == 0) ? l->row : l->row + (8 - l->row % 8);
    int src2_w = (r->col % 8 == 0) ? r->col : r->col + (8 - r->col % 8);
    int src2_h = (r->row % 8 == 0) ? r->row : r->row + (8 - r->row % 8);
    /* src1: l, src2: r */
    (*dst) = create(l->row, r->col);

    for (int i = 0; i < src1_h; i += 8) {
        for (int j = 0; j < src2_w; j += 8) {
            __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
                    ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

            __m256i ymm16 = _mm256_setzero_si256();
            __m256i ymm17 = _mm256_setzero_si256();
            __m256i ymm18 = _mm256_setzero_si256();
            __m256i ymm19 = _mm256_setzero_si256();
            __m256i ymm20 = _mm256_setzero_si256();
            __m256i ymm21 = _mm256_setzero_si256();
            __m256i ymm22 = _mm256_setzero_si256();
            __m256i ymm23 = _mm256_setzero_si256();

            for (int k = 0; k < src2_h; k += 8) {
                // load eight rows from source 2
                ymm0 = _mm256_loadu_si256((__m256i *) &(PRIV(r)->values[k + 0][j]));
                ymm1 = _mm256_loadu_si256((__m256i *) &(PRIV(r)->values[k + 1][j]));
                ymm2 = _mm256_loadu_si256((__m256i *) &(PRIV(r)->values[k + 2][j]));
                ymm3 = _mm256_loadu_si256((__m256i *) &(PRIV(r)->values[k + 3][j]));
                ymm4 = _mm256_loadu_si256((__m256i *) &(PRIV(r)->values[k + 4][j]));
                ymm5 = _mm256_loadu_si256((__m256i *) &(PRIV(r)->values[k + 5][j]));
                ymm6 = _mm256_loadu_si256((__m256i *) &(PRIV(r)->values[k + 6][j]));
                ymm7 = _mm256_loadu_si256((__m256i *) &(PRIV(r)->values[k + 7][j]));

                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(PRIV(l)->values[i + 0][k + 0]);
                ymm9 = _mm256_set1_epi32(PRIV(l)->values[i + 0][k + 1]);
                ymm10 = _mm256_set1_epi32(PRIV(l)->values[i + 0][k + 2]);
                ymm11 = _mm256_set1_epi32(PRIV(l)->values[i + 0][k + 3]);
                ymm12 = _mm256_set1_epi32(PRIV(l)->values[i + 0][k + 4]);
                ymm13 = _mm256_set1_epi32(PRIV(l)->values[i + 0][k + 5]);
                ymm14 = _mm256_set1_epi32(PRIV(l)->values[i + 0][k + 6]);
                ymm15 = _mm256_set1_epi32(PRIV(l)->values[i + 0][k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm16 = _mm256_add_epi32(ymm16, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(PRIV(l)->values[i + 1][k + 0]);
                ymm9 = _mm256_set1_epi32(PRIV(l)->values[i + 1][k + 1]);
                ymm10 = _mm256_set1_epi32(PRIV(l)->values[i + 1][k + 2]);
                ymm11 = _mm256_set1_epi32(PRIV(l)->values[i + 1][k + 3]);
                ymm12 = _mm256_set1_epi32(PRIV(l)->values[i + 1][k + 4]);
                ymm13 = _mm256_set1_epi32(PRIV(l)->values[i + 1][k + 5]);
                ymm14 = _mm256_set1_epi32(PRIV(l)->values[i + 1][k + 6]);
                ymm15 = _mm256_set1_epi32(PRIV(l)->values[i + 1][k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm17 = _mm256_add_epi32(ymm17, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(PRIV(l)->values[i + 2][k + 0]);
                ymm9 = _mm256_set1_epi32(PRIV(l)->values[i + 2][k + 1]);
                ymm10 = _mm256_set1_epi32(PRIV(l)->values[i + 2][k + 2]);
                ymm11 = _mm256_set1_epi32(PRIV(l)->values[i + 2][k + 3]);
                ymm12 = _mm256_set1_epi32(PRIV(l)->values[i + 2][k + 4]);
                ymm13 = _mm256_set1_epi32(PRIV(l)->values[i + 2][k + 5]);
                ymm14 = _mm256_set1_epi32(PRIV(l)->values[i + 2][k + 6]);
                ymm15 = _mm256_set1_epi32(PRIV(l)->values[i + 2][k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm18 = _mm256_add_epi32(ymm18, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(PRIV(l)->values[i + 3][k + 0]);
                ymm9 = _mm256_set1_epi32(PRIV(l)->values[i + 3][k + 1]);
                ymm10 = _mm256_set1_epi32(PRIV(l)->values[i + 3][k + 2]);
                ymm11 = _mm256_set1_epi32(PRIV(l)->values[i + 3][k + 3]);
                ymm12 = _mm256_set1_epi32(PRIV(l)->values[i + 3][k + 4]);
                ymm13 = _mm256_set1_epi32(PRIV(l)->values[i + 3][k + 5]);
                ymm14 = _mm256_set1_epi32(PRIV(l)->values[i + 3][k + 6]);
                ymm15 = _mm256_set1_epi32(PRIV(l)->values[i + 3][k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm19 = _mm256_add_epi32(ymm19, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(PRIV(l)->values[i + 4][k + 0]);
                ymm9 = _mm256_set1_epi32(PRIV(l)->values[i + 4][k + 1]);
                ymm10 = _mm256_set1_epi32(PRIV(l)->values[i + 4][k + 2]);
                ymm11 = _mm256_set1_epi32(PRIV(l)->values[i + 4][k + 3]);
                ymm12 = _mm256_set1_epi32(PRIV(l)->values[i + 4][k + 4]);
                ymm13 = _mm256_set1_epi32(PRIV(l)->values[i + 4][k + 5]);
                ymm14 = _mm256_set1_epi32(PRIV(l)->values[i + 4][k + 6]);
                ymm15 = _mm256_set1_epi32(PRIV(l)->values[i + 4][k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm20 = _mm256_add_epi32(ymm20, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(PRIV(l)->values[i + 5][k + 0]);
                ymm9 = _mm256_set1_epi32(PRIV(l)->values[i + 5][k + 1]);
                ymm10 = _mm256_set1_epi32(PRIV(l)->values[i + 5][k + 2]);
                ymm11 = _mm256_set1_epi32(PRIV(l)->values[i + 5][k + 3]);
                ymm12 = _mm256_set1_epi32(PRIV(l)->values[i + 5][k + 4]);
                ymm13 = _mm256_set1_epi32(PRIV(l)->values[i + 5][k + 5]);
                ymm14 = _mm256_set1_epi32(PRIV(l)->values[i + 5][k + 6]);
                ymm15 = _mm256_set1_epi32(PRIV(l)->values[i + 5][k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm21 = _mm256_add_epi32(ymm21, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(PRIV(l)->values[i + 6][k + 0]);
                ymm9 = _mm256_set1_epi32(PRIV(l)->values[i + 6][k + 1]);
                ymm10 = _mm256_set1_epi32(PRIV(l)->values[i + 6][k + 2]);
                ymm11 = _mm256_set1_epi32(PRIV(l)->values[i + 6][k + 3]);
                ymm12 = _mm256_set1_epi32(PRIV(l)->values[i + 6][k + 4]);
                ymm13 = _mm256_set1_epi32(PRIV(l)->values[i + 6][k + 5]);
                ymm14 = _mm256_set1_epi32(PRIV(l)->values[i + 6][k + 6]);
                ymm15 = _mm256_set1_epi32(PRIV(l)->values[i + 6][k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm22 = _mm256_add_epi32(ymm22, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(PRIV(l)->values[i + 7][k + 0]);
                ymm9 = _mm256_set1_epi32(PRIV(l)->values[i + 7][k + 1]);
                ymm10 = _mm256_set1_epi32(PRIV(l)->values[i + 7][k + 2]);
                ymm11 = _mm256_set1_epi32(PRIV(l)->values[i + 7][k + 3]);
                ymm12 = _mm256_set1_epi32(PRIV(l)->values[i + 7][k + 4]);
                ymm13 = _mm256_set1_epi32(PRIV(l)->values[i + 7][k + 5]);
                ymm14 = _mm256_set1_epi32(PRIV(l)->values[i + 7][k + 6]);
                ymm15 = _mm256_set1_epi32(PRIV(l)->values[i + 7][k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm23 = _mm256_add_epi32(ymm23, ymm8);
            }
            _mm256_storeu_si256((__m256i *) &(PRIV(*dst)->values[i + 0][j]), ymm16);
            _mm256_storeu_si256((__m256i *) &(PRIV(*dst)->values[i + 1][j]), ymm17);
            _mm256_storeu_si256((__m256i *) &(PRIV(*dst)->values[i + 2][j]), ymm18);
            _mm256_storeu_si256((__m256i *) &(PRIV(*dst)->values[i + 3][j]), ymm19);
            _mm256_storeu_si256((__m256i *) &(PRIV(*dst)->values[i + 4][j]), ymm20);
            _mm256_storeu_si256((__m256i *) &(PRIV(*dst)->values[i + 5][j]), ymm21);
            _mm256_storeu_si256((__m256i *) &(PRIV(*dst)->values[i + 6][j]), ymm22);
            _mm256_storeu_si256((__m256i *) &(PRIV(*dst)->values[i + 7][j]), ymm23);
        }
    }
    return true;
}

MatrixAlgo AVXMatrixProvider = {
    .create = create,
    .assign = assign,
    .equal = equal,
    .mul = avx_mul,
    .display = display,
};
