#include "matrix.h"
#include <stdlib.h>

struct naive_priv {
    int **values;
};

#define PRIV(x) \
    ((struct naive_priv *) ((x)->priv))

static Matrix *create(int row, int col)
{
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    m->row = row;
    m->col = col;
    m->priv = malloc(sizeof(struct naive_priv));
    PRIV(m)->values = (int **) malloc(row * sizeof(int *));
    for(int i = 0; i < row; i++)
        PRIV(m)->values[i] = (int *) calloc(col, sizeof(int));
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

static bool mul(Matrix **dst, const Matrix *l, const Matrix *r)
{
    if(l->col != r->row) return false;
    (*dst) = create(l->row, r->col);
    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < r->col; j++)
            for (int k = 0; k < l->col; k++)
                PRIV(*dst)->values[i][j] +=
                    PRIV(l)->values[i][k] * PRIV(r)->values[k][j];
    return true;
}

MatrixAlgo NaiveMatrixProvider = {
    .create = create,
    .assign = assign,
    .equal = equal,
    .mul = mul,
};
