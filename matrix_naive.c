#include "matrix.h"
#include <stdlib.h>

struct naive_priv {
    int *values;
};

#define PRIV(x) \
    ((struct naive_priv *) ((x)->priv))

static Matrix *create(int size)
{
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    m->size = size;
    m->priv = malloc(sizeof(struct naive_priv));
    PRIV(m)->values = (int *) malloc(size * size * sizeof(int));
    return m;
}

static void assign(Matrix *thiz, int *data)
{
    int size = thiz->size;
    for (int i = 0; i < size * size; i++)
        PRIV(thiz)->values[i] = *(data + i);
}

static bool equal(const Matrix *l, const Matrix *r)
{
    int size = l->size;
    for (int i = 0; i < size * size; i++)
        if (PRIV(l)->values[i] != PRIV(r)->values[i])
            return false;
    return true;
}

bool mul(Matrix *dst, const Matrix *l, const Matrix *r)
{
    int size = l->size;
    if(l->size != r->size) return false;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                PRIV(dst)->values[i * size + j] += PRIV(l)->values[i * size + k] *
                                                   PRIV(r)->values[k * size + j];
    return true;
}

MatrixAlgo NaiveMatrixProvider = {
    .create = create,
    .assign = assign,
    .equal = equal,
    .mul = mul,
};
