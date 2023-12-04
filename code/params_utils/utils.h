#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "../params_utils/parameters.h"


// define euler number needed for sigmoid function
#define EULER_NUMBER 2.71828


// initializes a matrix with shape: (row1, column1)
// used to initialize W_i, W_f, W_c, W_o and U_i, U_f, U_c, U_o Matrices 
void initialize_matrix(double** matrix, int row, int column, double initial_weight){
    int i, j;

    for (i = 0; i < row; i++){
        for(j = 0; j < column; j++){
            matrix[i][j] = initial_weight;
        }
    }
}

// initializes a vector with size: vec_size
// used to initialize b_i, b_f, b_c, b_o vectors
void initialize_vector(double* vector, int vec_size, double initial_weight){
    int i;
    for (i = 0; i < vec_size; i++){
            vector[i] = initial_weight;
        }
}

// multiplies two matrices together
// mat1 shape: (row1, column1)
// mat2 shape: (row2, column2)
double** matrix_mult(double** mat1, int row1, int column1, double** mat2, int row2, int column2){
    int i, j, k;

    double** mult;
    mult = malloc(sizeof(double*) * row1); 
    for(i = 0; i < row1; i++) { 
        mult[i] = malloc(sizeof(double*) * column2);
    }
    
    for (i = 0; i < row1; i++) { 
        for (j = 0; j < column2; j++) {
            mult[i][j] = 0;
 
            for (k = 0; k < row2; k++) {
                mult[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return mult;
}

// sum of two matrices with shape: (row, column)
double** matrix_sum_2(double** matrix1, double** matrix2, int row, int column){
    int i, j;

    double** sum;
    sum = malloc(sizeof(double*) * row);
    for(i = 0; i < row; i++) {
        sum[i] = malloc(sizeof(double*) * column);
    }
 
    for(i = 0; i < row; i++){
        for(j = 0; j < column; j++){
            sum[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    return sum;
}

// sum of three matrices with shape: (row, column)
double** matrix_sum_3(double** matrix1, double** matrix2, double** matrix3, int row, int column){
    int i, j;

    double** sum;
    sum = malloc(sizeof(double*) * row);
    for(i = 0; i < row; i++) {
        sum[i] = malloc(sizeof(double*) * column);
    }
 
    for(i = 0; i < row; i++){
        for(j = 0; j < column; j++){
            sum[i][j] = matrix1[i][j] + matrix2[i][j] + matrix3[i][j];
        }
    }
    return sum;
}

// broadcast a vector with size vec_size to a matrix with shape: (row, vec_size)
double** broadcast_vector_to_matrix(double* vector, int row, int vec_size){
    int i, j;

    double** matrix;
    matrix = malloc(sizeof(double*) * row);
    for(i = 0; i < row; i++) {
        matrix[i] = malloc(sizeof(double*) * vec_size);
    }
    
    for (i = 0; i < row; i++) { 
        for (j = 0; j < vec_size; j++) { 
            matrix[i][j] = vector[j];
        }
    }
    return matrix;
}

// sigmoid function for a number
double sigmoid(double n) {
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}

// sigmoid function broadcasted to a matrix with shape: (batch_size, hidden_size)
double** matrix_sigmoid(double** matrix, int row, int column){
    int i, j;

    double** sigmoid_matrix;
    sigmoid_matrix = malloc(sizeof(double*) * row);
    for(i = 0; i < row; i++) {
        sigmoid_matrix[i] = malloc(sizeof(double*) * column);
    }
 
    for(i = 0; i < row; i++){
        for(j = 0; j < column; j++){
            sigmoid_matrix[i][j] = sigmoid(matrix[i][j]);
        }
    }
    return sigmoid_matrix;
}

// tanh function broadcasted to a matrix
double** matrix_tanh(double** matrix, int row, int column){
    int i, j;

    double** tanh_matrix;
    tanh_matrix = malloc(sizeof(double*) * row);
    for(i = 0; i < row; i++) {
        tanh_matrix[i] = malloc(sizeof(double*) * column);
    }
 
    for(i = 0; i < row; i++){
        for(j = 0; j < column; j++){
            tanh_matrix[i][j] = tanh(matrix[i][j]);
        }
    }
    return tanh_matrix;
}

// matrix product of two matrices with shape: (batch_size, hidden_size)
double** matrix_product(double** matrix1, double** matrix2, int row, int column){
    int i, j;

    double** product;
    product = malloc(sizeof(double*) * row); 
    for(i = 0; i < row; i++) { 
        product[i] = malloc(sizeof(double*) * column);
    }
    
    for (i = 0; i < row; i++) { 
        for (j = 0; j < column; j++) { 
            product[i][j] = matrix1[i][j] * matrix2[i][j];
        }
    }
    return product;
}

// average of two matrices with shape: (row, column)
double** matrix_avg_2(double** matrix1, double** matrix2, int row, int column){
    int i, j;

    double** avg;
    avg = malloc(sizeof(double*) * row);
    for(i = 0; i < row; i++) {
        avg[i] = malloc(sizeof(double*) * column);
    }
 
    for(i = 0; i < row; i++){
        for(j = 0; j < column; j++){
            avg[i][j] = (matrix1[i][j] + matrix2[i][j]) / 2;
        }
    }
    return avg;
}

// concatenates matrices mat1 and mat2 with shape: (row, column)
// output matrix has shape: (row, 2*column)
double** matrix_concat_2(double** mat1, double** mat2, int row, int column){
    int i, j, k;

    double** concatenated;
    concatenated = malloc(sizeof(double*) * row);
    for(i = 0; i < row; i++) { 
        concatenated[i] = malloc(sizeof(double*) * 2*column);
    }

    for (i = 0; i < row; i++){
        for (j = 0; j < column; j++){
            concatenated[i][j] = mat1[i][j];
            concatenated[i][j+column] = mat2[i][j];
        }
    }

    return concatenated;
}
        
// subtracts two matrices with shape (row, column)
double** matrix_subtract_2(double** matrix1, double** matrix2, int row, int column){
    int i, j;
    double** subtract;
    subtract = malloc(sizeof(double*) * row);
     
    for(i = 0; i < row; i++) {
        subtract[i] = malloc(sizeof(double*) * column);
    }
 
    for(i = 0; i < row; i++){
        for(j = 0; j < column; j++){
            subtract[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
    return subtract;
}

// creates a matrix with shape (row, column) and assigns all elements to one
double** matrix_one(int row, int column){
    int i, j;
    double** matrix_one;

    matrix_one = malloc(sizeof(double*) * row); 

    for(i = 0; i < row; i++) { 
        matrix_one[i] = malloc(sizeof(double*) * column);
    }

    for (i = 0; i < row; i++){
        for (j = 0; j < column; j++){
            matrix_one[i][j] = 1;
        }
    }
    return matrix_one;
}