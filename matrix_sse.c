#include "matrix.h"
#include <stdlib.h>
#include <immintrin.h>

struct sse_priv {
    int **values;
};

#define PRIV(x) \
    ((struct sse_priv *) ((x)->priv))

static Matrix *create(int row, int col)
{
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    int actual_row = (row % 4 == 0) ? row : row + (4 - row % 4);
    int actual_col = (col % 4 == 0) ? col : col + (4 - col % 4);
    m->row = row;
    m->col = col;
    m->priv = malloc(sizeof(struct sse_priv));
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
        for (int j = 0; j < l->col; j ++)
            if (PRIV(l)->values[i][j] != PRIV(r)->values[i][j])
                return false;
    return true;
}

static bool sse_mul(Matrix **dst, const Matrix *l, const Matrix *r)
{
    if(l->col != r->row) return false;
    int src1_h = (l->row % 4 == 0) ? l->row : l->row + (4 - l->row % 4);
    int src2_w = (r->col % 4 == 0) ? r->col : r->col + (4 - r->col % 4);
    /* src1: l, src2: r */
    (*dst) = create(l->row, r->col);

    for (int i = 0; i < src1_h; i += 4) {
        for (int j = 0; j < src2_w; j+= 4) {
            __m128i des0 = _mm_setzero_si128 ();
            __m128i des1 = _mm_setzero_si128 ();
            __m128i des2 = _mm_setzero_si128 ();
            __m128i des3 = _mm_setzero_si128 ();
            for (int k = 0; k < src2_w; k += 4) {
                __m128i I0 = _mm_load_si128((__m128i *)&(PRIV(l)->values[i + 0][k]));
                __m128i I1 = _mm_load_si128((__m128i *)&(PRIV(l)->values[i + 1][k]));
                __m128i I2 = _mm_load_si128((__m128i *)&(PRIV(l)->values[i + 2][k]));
                __m128i I3 = _mm_load_si128((__m128i *)&(PRIV(l)->values[i + 3][k]));
                __m128i I4 = _mm_set_epi32 (PRIV(r)->values[k + 3][j], PRIV(r)->values[k + 2][j],
                                            PRIV(r)->values[k + 1][j], PRIV(r)->values[k + 0][j]);
                __m128i I5 = _mm_set_epi32 (PRIV(r)->values[k + 3][j + 1], PRIV(r)->values[k + 2][j + 1],
                                            PRIV(r)->values[k + 1][j + 1], PRIV(r)->values[k + 0][j + 1]);
                __m128i I6 = _mm_set_epi32 (PRIV(r)->values[k + 3][j + 2], PRIV(r)->values[k + 2][j + 2],
                                            PRIV(r)->values[k + 1][j + 2], PRIV(r)->values[k + 0][j + 2]);
                __m128i I7 = _mm_set_epi32 (PRIV(r)->values[k + 3][j + 3], PRIV(r)->values[k + 2][j + 3],
                                            PRIV(r)->values[k + 1][j + 3], PRIV(r)->values[k + 0][j + 3]);

                __m128i T0 = _mm_mullo_epi32(I0, I4);
                __m128i T1 = _mm_mullo_epi32(I0, I5);
                __m128i T2 = _mm_mullo_epi32(I0, I6);
                __m128i T3 = _mm_mullo_epi32(I0, I7);

                __m128i T4 = _mm_mullo_epi32(I1, I4);
                __m128i T5 = _mm_mullo_epi32(I1, I5);
                __m128i T6 = _mm_mullo_epi32(I1, I6);
                __m128i T7 = _mm_mullo_epi32(I1, I7);

                __m128i T8 = _mm_mullo_epi32(I2, I4);
                __m128i T9 = _mm_mullo_epi32(I2, I5);
                __m128i T10 = _mm_mullo_epi32(I2, I6);
                __m128i T11 = _mm_mullo_epi32(I2, I7);

                __m128i T12 = _mm_mullo_epi32(I3, I4);
                __m128i T13 = _mm_mullo_epi32(I3, I5);
                __m128i T14 = _mm_mullo_epi32(I3, I6);
                __m128i T15 = _mm_mullo_epi32(I3, I7);

                __m128i T16 = _mm_unpacklo_epi32(T0, T1);
                __m128i T17 = _mm_unpacklo_epi32(T2, T3);
                __m128i T18 = _mm_unpackhi_epi32(T0, T1);
                __m128i T19 = _mm_unpackhi_epi32(T2, T3);

                __m128i T20 = _mm_unpacklo_epi64(T16, T17);
                __m128i T21 = _mm_unpackhi_epi64(T16, T17);
                __m128i T22 = _mm_unpacklo_epi64(T18, T19);
                __m128i T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des0 = _mm_add_epi32(T20, des0);

                T16 = _mm_unpacklo_epi32(T4, T5);
                T17 = _mm_unpacklo_epi32(T6, T7);
                T18 = _mm_unpackhi_epi32(T4, T5);
                T19 = _mm_unpackhi_epi32(T6, T7);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des1 = _mm_add_epi32(T20, des1);

                T16 = _mm_unpacklo_epi32(T8, T9);
                T17 = _mm_unpacklo_epi32(T10, T11);
                T18 = _mm_unpackhi_epi32(T8, T9);
                T19 = _mm_unpackhi_epi32(T10, T11);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des2 = _mm_add_epi32(T20, des2);

                T16 = _mm_unpacklo_epi32(T12, T13);
                T17 = _mm_unpacklo_epi32(T14, T15);
                T18 = _mm_unpackhi_epi32(T12, T13);
                T19 = _mm_unpackhi_epi32(T14, T15);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des3 = _mm_add_epi32(T20, des3);
            }

            _mm_store_si128((__m128i *)&(PRIV(*dst)->values[i + 0][j]), des0);
            _mm_store_si128((__m128i *)&(PRIV(*dst)->values[i + 1][j]), des1);
            _mm_store_si128((__m128i *)&(PRIV(*dst)->values[i + 2][j]), des2);
            _mm_store_si128((__m128i *)&(PRIV(*dst)->values[i + 3][j]), des3);
        }
    }
    return true;
}

static bool sse_prefetch_mul(Matrix **dst, const Matrix *l, const Matrix *r)
{
    if(l->col != r->row) return false;
    int src1_h = (l->row % 4 == 0) ? l->row : l->row + (4 - l->row % 4);
    int src2_w = (r->col % 4 == 0) ? r->col : r->col + (4 - r->col % 4);
    /* src1: l, src2: r */
    (*dst) = create(l->row, r->col);

    for (int i = 0; i < src1_h; i += 4) {
        for (int j = 0; j < src2_w; j+= 4) {
            __m128i des0 = _mm_setzero_si128 ();
            __m128i des1 = _mm_setzero_si128 ();
            __m128i des2 = _mm_setzero_si128 ();
            __m128i des3 = _mm_setzero_si128 ();
            for (int k = 0; k < src2_w; k += 4) {
#define SSE_PFDIST 8
                _mm_prefetch(&(PRIV(r)->values[k + SSE_PFDIST + 0][j]), _MM_HINT_T1);
                _mm_prefetch(&(PRIV(r)->values[k + SSE_PFDIST + 1][j]), _MM_HINT_T1);
                _mm_prefetch(&(PRIV(r)->values[k + SSE_PFDIST + 2][j]), _MM_HINT_T1);
                _mm_prefetch(&(PRIV(r)->values[k + SSE_PFDIST + 3][j]), _MM_HINT_T1);

                __m128i I0 = _mm_load_si128((__m128i *)&(PRIV(l)->values[i + 0][k]));
                __m128i I1 = _mm_load_si128((__m128i *)&(PRIV(l)->values[i + 1][k]));
                __m128i I2 = _mm_load_si128((__m128i *)&(PRIV(l)->values[i + 2][k]));
                __m128i I3 = _mm_load_si128((__m128i *)&(PRIV(l)->values[i + 3][k]));
                __m128i I4 = _mm_set_epi32 (PRIV(r)->values[k + 3][j], PRIV(r)->values[k + 2][j],
                                            PRIV(r)->values[k + 1][j], PRIV(r)->values[k + 0][j]);
                __m128i I5 = _mm_set_epi32 (PRIV(r)->values[k + 3][j + 1], PRIV(r)->values[k + 2][j + 1],
                                            PRIV(r)->values[k + 1][j + 1], PRIV(r)->values[k + 0][j + 1]);
                __m128i I6 = _mm_set_epi32 (PRIV(r)->values[k + 3][j + 2], PRIV(r)->values[k + 2][j + 2],
                                            PRIV(r)->values[k + 1][j + 2], PRIV(r)->values[k + 0][j + 2]);
                __m128i I7 = _mm_set_epi32 (PRIV(r)->values[k + 3][j + 3], PRIV(r)->values[k + 2][j + 3],
                                            PRIV(r)->values[k + 1][j + 3], PRIV(r)->values[k + 0][j + 3]);

                __m128i T0 = _mm_mullo_epi32(I0, I4);
                __m128i T1 = _mm_mullo_epi32(I0, I5);
                __m128i T2 = _mm_mullo_epi32(I0, I6);
                __m128i T3 = _mm_mullo_epi32(I0, I7);

                __m128i T4 = _mm_mullo_epi32(I1, I4);
                __m128i T5 = _mm_mullo_epi32(I1, I5);
                __m128i T6 = _mm_mullo_epi32(I1, I6);
                __m128i T7 = _mm_mullo_epi32(I1, I7);

                __m128i T8 = _mm_mullo_epi32(I2, I4);
                __m128i T9 = _mm_mullo_epi32(I2, I5);
                __m128i T10 = _mm_mullo_epi32(I2, I6);
                __m128i T11 = _mm_mullo_epi32(I2, I7);

                __m128i T12 = _mm_mullo_epi32(I3, I4);
                __m128i T13 = _mm_mullo_epi32(I3, I5);
                __m128i T14 = _mm_mullo_epi32(I3, I6);
                __m128i T15 = _mm_mullo_epi32(I3, I7);

                __m128i T16 = _mm_unpacklo_epi32(T0, T1);
                __m128i T17 = _mm_unpacklo_epi32(T2, T3);
                __m128i T18 = _mm_unpackhi_epi32(T0, T1);
                __m128i T19 = _mm_unpackhi_epi32(T2, T3);

                __m128i T20 = _mm_unpacklo_epi64(T16, T17);
                __m128i T21 = _mm_unpackhi_epi64(T16, T17);
                __m128i T22 = _mm_unpacklo_epi64(T18, T19);
                __m128i T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des0 = _mm_add_epi32(T20, des0);

                T16 = _mm_unpacklo_epi32(T4, T5);
                T17 = _mm_unpacklo_epi32(T6, T7);
                T18 = _mm_unpackhi_epi32(T4, T5);
                T19 = _mm_unpackhi_epi32(T6, T7);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des1 = _mm_add_epi32(T20, des1);

                T16 = _mm_unpacklo_epi32(T8, T9);
                T17 = _mm_unpacklo_epi32(T10, T11);
                T18 = _mm_unpackhi_epi32(T8, T9);
                T19 = _mm_unpackhi_epi32(T10, T11);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des2 = _mm_add_epi32(T20, des2);

                T16 = _mm_unpacklo_epi32(T12, T13);
                T17 = _mm_unpacklo_epi32(T14, T15);
                T18 = _mm_unpackhi_epi32(T12, T13);
                T19 = _mm_unpackhi_epi32(T14, T15);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des3 = _mm_add_epi32(T20, des3);
            }

            _mm_store_si128((__m128i *)&(PRIV(*dst)->values[i + 0][j]), des0);
            _mm_store_si128((__m128i *)&(PRIV(*dst)->values[i + 1][j]), des1);
            _mm_store_si128((__m128i *)&(PRIV(*dst)->values[i + 2][j]), des2);
            _mm_store_si128((__m128i *)&(PRIV(*dst)->values[i + 3][j]), des3);
        }
    }
    return true;
}

MatrixAlgo SSEMatrixProvider = {
    .create = create,
    .assign = assign,
    .equal = equal,
    .mul = sse_mul,
};

MatrixAlgo prefetchSSEMatrixProvider = {
    .create = create,
    .assign = assign,
    .equal = equal,
    .mul = sse_prefetch_mul,
};
