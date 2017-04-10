#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdbool.h>

/* predefined shortcut */
#define DECLARE_MATRIX(col, row) \
    typedef struct { int values[col][row]; } Mat ## col ## x ## row

typedef struct {
    int row, col;
    void *priv;
} Matrix;

typedef struct {
    Matrix *(*create)(int row, int col);
    void (*assign)(Matrix *thiz, int *data);
    bool (*equal)(const Matrix *l, const Matrix *r);
    bool (*mul)(Matrix **dst, const Matrix *l, const Matrix *r);
    void (*display)(Matrix *thiz);
} MatrixAlgo;

/* Available matrix providers */
extern MatrixAlgo NaiveMatrixProvider;
extern MatrixAlgo SSEMatrixProvider;
extern MatrixAlgo prefetchSSEMatrixProvider;
extern MatrixAlgo AVXMatrixProvider;
extern MatrixAlgo prefetchAVXMatrixProvider;

#endif /* MATRIX_H_ */
