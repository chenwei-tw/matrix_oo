#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdbool.h>

typedef struct {
    int row, col;
    float **priv;
} Matrix;

typedef struct {
    Matrix *(*create)(int row, int col);
    bool (*assign)(Matrix *thiz, float **data, int row, int col);
    bool (*equal)(const Matrix *l, const Matrix *r);
    Matrix *(*mul)(const Matrix *l, const Matrix *r);
} MatrixAlgo;

/* Available matrix providers */
extern MatrixAlgo NaiveMatrixProvider;

#endif /* MATRIX_H_ */
