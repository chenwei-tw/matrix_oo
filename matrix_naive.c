#include "matrix.h"
#include <stdlib.h>

static Matrix *create(int row, int col)
{
    Matrix *s = (Matrix *) malloc(sizeof(Matrix));
    s->row = row;
    s->col = col;
    s->priv = (float **) malloc(row * sizeof(float));
    for(int i = 0; i < row; i++)
        s->priv[i] = (float *) malloc(col * sizeof(float));
    return s;
}

static bool assign(Matrix *thiz, float **data, int row, int col )
{
    if(thiz->row != row || thiz->col != col) return false;
    for (int i = 0; i < thiz->row; i++)
        for (int j = 0; j < thiz->col; j++)
            thiz->priv[i][j] = *((float *) data + i * col + j);
    return true;
}

static const float epsilon = 1 / 10000.0;

static bool equal(const Matrix *l, const Matrix *r)
{
    if(l->row != r->row || l->col != l->col) return false;
    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < l->col; j++)
            if (l->priv[i][j] + epsilon < r->priv[i][j] ||
                    r->priv[i][j] + epsilon < l->priv[i][j])
                return false;
    return true;
}

static Matrix *mul(const Matrix *l, const Matrix *r)
{
    if(l->col != r->row) return false;
    Matrix *dst = create(l->row, r->col);
    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < r->col; j++)
            for (int k = 0; k < r->row; k++)
                dst->priv[i][j] += l->priv[i][k] *
                                   r->priv[k][j];
    return dst;
}

MatrixAlgo NaiveMatrixProvider = {
    .create = create,
    .assign = assign,
    .equal = equal,
    .mul = mul,
};
