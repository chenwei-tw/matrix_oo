#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdbool.h>

typedef struct {
    int row, col;
    void *priv;
} Matrix;

typedef struct {
    Matrix *(*create)(int row, int col);
    void (*assign)(Matrix *thiz, int *data);
    bool (*equal)(const Matrix *l, const Matrix *r);
    bool (*mul)(Matrix **dst, const Matrix *l, const Matrix *r);
} MatrixAlgo;

/* Available matrix providers */
extern MatrixAlgo NaiveMatrixProvider;

#endif /* MATRIX_H_ */
