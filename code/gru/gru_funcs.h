#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<time.h>
#include <string.h>

#include "../params_utils/parameters.h"
#include "../params_utils/utils.h"


double** forward_gru(double x[batch_size][sequence_size][input_size], double** W_z,
double** U_z, double** matrix_bz, double** W_r, double** U_r, double** matrix_br,
double** W_h, double** U_h, double** matrix_bh, double** h_t_param, double initial_weight){
    int i, j, k;

    // x_t is one batch of our input x in each iteration
    double** x_t; // shape: (batch_size, input_size)
    x_t = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        x_t[i] = malloc(sizeof(double*) * input_size);
    }

    // a matrix of ones with shape: (batch_size, hidden_size)
    double** ones;
    ones = matrix_one(batch_size, hidden_size);

    // *** all matrices defined below have shape (batch_size, hidden_size) ***
    // used to compute z_t
    double** mult_xt_Wz; 
    double** mult_ht_Uz; 
    double** sum_z;
    double** z_t; 

    // used to compute r_t
    double** mult_xt_Wr; 
    double** mult_ht_Ur;
    double** sum_r;
    double** r_t; 

    // used to compute h_tilde
    double** mult_xt_Wh; 
    double** prod_r_ht;
    double** mult_prod_r_ht_and_Uh;
    double** sum_h_tilde;
    double** h_tilde; 

    // used to compute h_t
    double** prod_z_ht;
    double** one_minus_zt;
    double** prod_one_minus_zt_and_h_tilde;

    
    // create matrix h_t and initialize it with the values from the input of function
    double** h_t;
    h_t = malloc(sizeof(double*) * batch_size);
    for(i = 0; i < batch_size; i++) { 
        h_t[i] = malloc(sizeof(double*) * hidden_size);
    }

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            h_t[i][j] = h_t_param[i][j];
        }
    }


    // iterate on each sequence and compute h_t and c_t
    for (j = 0; j < sequence_size; j++){
        for (i = 0; i < batch_size; i++){
            for (k = 0; k < input_size; k++){
                x_t[i][k] = x[i][j][k];
            }
        }

        // compute z_t (update gate)
        mult_xt_Wz = matrix_mult(x_t, batch_size, input_size, W_z, input_size, hidden_size);
        mult_ht_Uz = matrix_mult(h_t, batch_size, hidden_size, U_z, hidden_size, hidden_size);
        sum_z = matrix_sum_3(mult_xt_Wz, mult_ht_Uz, matrix_bz, batch_size, hidden_size);
        z_t = matrix_sigmoid(sum_z, batch_size, hidden_size);

        // compute r_t (reset gate)
        mult_xt_Wr = matrix_mult(x_t, batch_size, input_size, W_r, input_size, hidden_size);
        mult_ht_Ur = matrix_mult(h_t, batch_size, hidden_size, U_r, hidden_size, hidden_size);
        sum_r = matrix_sum_3(mult_xt_Wr, mult_ht_Ur, matrix_br, batch_size, hidden_size);
        r_t = matrix_sigmoid(sum_r, batch_size, hidden_size);

        // compute h_tilde (intermediate memory)
        mult_xt_Wh = matrix_mult(x_t, batch_size, input_size, W_h, input_size, hidden_size);
        prod_r_ht = matrix_product(r_t, h_t, batch_size, hidden_size);
        mult_prod_r_ht_and_Uh = matrix_mult(prod_r_ht, batch_size, hidden_size, U_h, hidden_size, hidden_size);
        sum_h_tilde = matrix_sum_3(mult_xt_Wh, mult_prod_r_ht_and_Uh, matrix_bh, batch_size, hidden_size);
        h_tilde = matrix_tanh(sum_h_tilde, batch_size, hidden_size);

        // compute h_t
        prod_z_ht = matrix_product(z_t, h_t, batch_size, hidden_size);
        one_minus_zt = matrix_subtract_2(ones, z_t, batch_size, hidden_size);
        prod_one_minus_zt_and_h_tilde = matrix_product(one_minus_zt, h_tilde, batch_size, hidden_size);
        h_t = matrix_sum_2(prod_z_ht, prod_one_minus_zt_and_h_tilde, batch_size, hidden_size);
    }

    return h_t;
} 

double** backward_gru(double x[batch_size][sequence_size][input_size], double** W_z,
double** U_z, double** matrix_bz, double** W_r, double** U_r, double** matrix_br,
double** W_h, double** U_h, double** matrix_bh, double** h_t_param, double initial_weight){
    int i, j, k;

    // x_t is one batch of our input x in each iteration
    double** x_t; // shape: (batch_size, input_size)
    x_t = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        x_t[i] = malloc(sizeof(double*) * input_size);
    }

    // a matrix of ones with shape: (batch_size, hidden_size)
    double** ones;
    ones = matrix_one(batch_size, hidden_size);

    // *** all matrices defined below have shape (batch_size, hidden_size) ***
    // used to compute z_t
    double** mult_xt_Wz; 
    double** mult_Uz_ht; 
    double** sum_z;
    double** z_t; 

    // used to compute r_t
    double** mult_xt_Wr; 
    double** mult_Ur_ht;
    double** sum_r;
    double** r_t; 

    // used to compute h_tilde
    double** mult_xt_Wh; 
    double** prod_r_ht;
    double** mult_prod_rh_and_Uh;
    double** sum_h_tilde;
    double** h_tilde; 

    // used to compute h_t
    double** prod_z_h;
    double** one_minus_zt;
    double** prod_one_minus_zt_and_h_tilde;

    
    // create matrix h_t and initialize it with the values from the input of function
    double** h_t;
    h_t = malloc(sizeof(double*) * batch_size);
    for(i = 0; i < batch_size; i++) { 
        h_t[i] = malloc(sizeof(double*) * hidden_size);
    }

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            h_t[i][j] = h_t_param[i][j];
        }
    }


    // iterate on each sequence and compute h_t and c_t
    for (j = sequence_size-1; j >= 0; j--){
        for (i = 0; i < batch_size; i++){
            for (k = 0; k < input_size; k++){
                x_t[i][k] = x[i][j][k];
            }
        }

        // compute z_t (update gate)
        mult_xt_Wz = matrix_mult(x_t, batch_size, input_size, W_z, input_size, hidden_size);
        mult_Uz_ht = matrix_mult(h_t, batch_size, hidden_size, U_z, hidden_size, hidden_size);
        sum_z = matrix_sum_3(mult_xt_Wz, mult_Uz_ht, matrix_bz, batch_size, hidden_size);
        z_t = matrix_sigmoid(sum_z, batch_size, hidden_size);

        // compute r_t (reset gate)
        mult_xt_Wr = matrix_mult(x_t, batch_size, input_size, W_r, input_size, hidden_size);
        mult_Ur_ht = matrix_mult(h_t, batch_size, hidden_size, U_r, hidden_size, hidden_size);
        sum_r = matrix_sum_3(mult_xt_Wr, mult_Ur_ht, matrix_br, batch_size, hidden_size);
        r_t = matrix_sigmoid(sum_r, batch_size, hidden_size);

        // compute h_tilde (intermediate memory)
        mult_xt_Wh = matrix_mult(x_t, batch_size, input_size, W_h, input_size, hidden_size);
        prod_r_ht = matrix_product(r_t, h_t, batch_size, hidden_size);
        mult_prod_rh_and_Uh = matrix_mult(prod_r_ht, batch_size, hidden_size, U_h, hidden_size, hidden_size);
        sum_h_tilde = matrix_sum_3(mult_xt_Wh, mult_prod_rh_and_Uh, matrix_bh, batch_size, hidden_size);
        h_tilde = matrix_tanh(sum_h_tilde, batch_size, hidden_size);

        // compute h_t
        prod_z_h = matrix_product(z_t, h_t, batch_size, hidden_size);
        one_minus_zt = matrix_subtract_2(ones, z_t, batch_size, hidden_size);
        prod_one_minus_zt_and_h_tilde = matrix_product(one_minus_zt, h_tilde, batch_size, hidden_size);
        h_t = matrix_sum_2(prod_z_h, prod_one_minus_zt_and_h_tilde, batch_size, hidden_size);
    }

    return h_t;
}

double** bi_gru(double** h_t_forward, double** h_t_backward, char merge_mode[]){
    int i, j;

    // add forward and backward h_t together
    if (strcmp(merge_mode, "sum") == 0){
        return matrix_sum_2(h_t_forward, h_t_backward, batch_size, hidden_size);
    }

    // multiply forward and backward h_t together
    else if (strcmp(merge_mode, "mult") == 0){
        return matrix_product(h_t_forward, h_t_backward, batch_size, hidden_size);
    }

    // take the average of forward and backward h_t 
    else if (strcmp(merge_mode, "avg") == 0) {
        return matrix_avg_2(h_t_forward, h_t_backward, batch_size, hidden_size);
    }

    // concatenate forward and backward h_t 
    else if (strcmp(merge_mode, "concat") == 0) {
        return matrix_concat_2(h_t_forward, h_t_backward, batch_size, hidden_size);
    }

    else{
        printf("values accepted for merge_mode are: sum, mult, avg, concat");
    }
}