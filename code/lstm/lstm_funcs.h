#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "../params_utils/parameters.h"
#include "../params_utils/utils.h"
        

// forward function of LSTM
// computes new values for h_t and c_t 
double** forward_lstm(double x[batch_size][sequence_size][input_size], double** W_i,
double** U_i, double** matrix_bi, double** W_f, double** U_f, double** matrix_bf,
double** W_c, double** U_c, double** matrix_bc, double** W_o, double** U_o,
double** matrix_bo, double** c_t_param, double** h_t_param, double initial_weight){

    int i, j, k;

    // x_t is one batch of our input x in each iteration
    double** x_t; // shape: (batch_size, input_size) 
    x_t = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        x_t[i] = malloc(sizeof(double*) * input_size);
    }

    // *** all matrices defined below have shape: (batch_size, hidden_size) ***
    // used to compute i_t (input gate)
    double** mult_xt_Wi;
    double** mult_ht_Ui;
    double** sum_i;
    double** i_t;

    // used to compute f_t (forget gate)
    double** mult_xt_Wf;
    double** mult_ht_Uf;
    double** sum_f;
    double** f_t;

    // used to compute g_t
    double** mult_xt_Wc;
    double** mult_ht_Uc;
    double** sum_c;
    double** g_t;

    // used to compute o_t (output gate)
    double** mult_xt_Wo;
    double** mult_ht_Uo;
    double** sum_o;
    double** o_t;

    // used to compute c_t (cell state)
    double** mult_ft_ct;
    double** mult_it_gt;

    // create matrix c_t and initialize it with the values from the input of function
    double** c_t;
    c_t = malloc(sizeof(double*) * batch_size);
    for(i = 0; i < batch_size; i++) { 
        c_t[i] = malloc(sizeof(double*) * hidden_size);
    }

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            c_t[i][j] = c_t_param[i][j];
        }
    }


    // used to compute h_t 
    double** tanh_ct;

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

        // compute i_t
        mult_xt_Wi = matrix_mult(x_t, batch_size, input_size, W_i, input_size, hidden_size);
        mult_ht_Ui = matrix_mult(h_t, batch_size, hidden_size, U_i, hidden_size, hidden_size);
        sum_i = matrix_sum_3(mult_xt_Wi, mult_ht_Ui, matrix_bi, batch_size, hidden_size);
        i_t = matrix_sigmoid(sum_i, batch_size, hidden_size);

        // compute f_t
        mult_xt_Wf = matrix_mult(x_t, batch_size, input_size, W_f, input_size, hidden_size);
        mult_ht_Uf = matrix_mult(h_t, batch_size, hidden_size, U_f, hidden_size, hidden_size);
        sum_f = matrix_sum_3(mult_xt_Wf, mult_ht_Uf, matrix_bf, batch_size, hidden_size);
        f_t = matrix_sigmoid(sum_f, batch_size, hidden_size);

        // compute g_t
        mult_xt_Wc = matrix_mult(x_t, batch_size, input_size, W_c, input_size, hidden_size);
        mult_ht_Uc = matrix_mult(h_t, batch_size, hidden_size, U_c, hidden_size, hidden_size);
        sum_c = matrix_sum_3(mult_xt_Wc, mult_ht_Uc, matrix_bc, batch_size, hidden_size);
        g_t = matrix_tanh(sum_c, batch_size, hidden_size);

        // compute o_t
        mult_xt_Wo = matrix_mult(x_t, batch_size, input_size, W_o, input_size, hidden_size);
        mult_ht_Uo = matrix_mult(h_t, batch_size, hidden_size, U_o, hidden_size, hidden_size);
        sum_o = matrix_sum_3(mult_xt_Wo, mult_ht_Uo, matrix_bo, batch_size, hidden_size);
        o_t = matrix_sigmoid(sum_o, batch_size, hidden_size);

        // compute c_t
        mult_ft_ct = matrix_product(f_t, c_t, batch_size, hidden_size);
        mult_it_gt = matrix_product(i_t, g_t, batch_size, hidden_size);
        c_t = matrix_sum_2(mult_ft_ct, mult_it_gt, batch_size, hidden_size);

        // compute h_t
        tanh_ct = matrix_tanh(c_t, batch_size, hidden_size);
        h_t = matrix_product(o_t, tanh_ct, batch_size, hidden_size);
    }

    return h_t;
}

// forward function of LSTM
// computes new values for h_t and c_t 
double** backward_lstm(double x[batch_size][sequence_size][input_size], double** W_i,
double** U_i, double** matrix_bi, double** W_f, double** U_f, double** matrix_bf,
double** W_c, double** U_c, double** matrix_bc, double** W_o, double** U_o,
double** matrix_bo, double** c_t_param, double** h_t_param, double initial_weight){

    int i, j, k;

    // x_t is one batch of our input x in each iteration
    double** x_t; // shape: (batch_size, input_size) 
    x_t = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        x_t[i] = malloc(sizeof(double*) * input_size);
    }

    // *** all matrices defined below have shape: (batch_size, hidden_size) ***
    // used to compute i_t (input gate)
    double** mult_xt_Wi;
    double** mult_ht_Ui;
    double** sum_i;
    double** i_t;

    // used to compute f_t (forget gate)
    double** mult_xt_Wf;
    double** mult_ht_Uf;
    double** sum_f;
    double** f_t;

    // used to compute g_t
    double** mult_xt_Wc;
    double** mult_ht_Uc;
    double** sum_c;
    double** g_t;

    // used to compute o_t (output gate)
    double** mult_xt_Wo;
    double** mult_ht_Uo;
    double** sum_o;
    double** o_t;

    // used to compute c_t (cell state)
    double** mult_ft_ct;
    double** mult_it_gt;

    // create matrix c_t and initialize it with the values from the input of function
    double** c_t;
    c_t = malloc(sizeof(double*) * batch_size);
    for(i = 0; i < batch_size; i++) { 
        c_t[i] = malloc(sizeof(double*) * hidden_size);
    }

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            c_t[i][j] = c_t_param[i][j];
        }
    }


    // used to compute h_t 
    double** tanh_ct;

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
    // *** only difference between forward_lstm and backward lstm
    // it iterates the input from end to start ***
    for (j = sequence_size-1; j >= 0; j--){
        for (i = 0; i < batch_size; i++){
            for (k = 0; k < input_size; k++){
                x_t[i][k] = x[i][j][k];
            }
        }

        // compute i_t
        mult_xt_Wi = matrix_mult(x_t, batch_size, input_size, W_i, input_size, hidden_size);
        mult_ht_Ui = matrix_mult(h_t, batch_size, hidden_size, U_i, hidden_size, hidden_size);
        sum_i = matrix_sum_3(mult_xt_Wi, mult_ht_Ui, matrix_bi, batch_size, hidden_size);
        i_t = matrix_sigmoid(sum_i, batch_size, hidden_size);

        // compute f_t
        mult_xt_Wf = matrix_mult(x_t, batch_size, input_size, W_f, input_size, hidden_size);
        mult_ht_Uf = matrix_mult(h_t, batch_size, hidden_size, U_f, hidden_size, hidden_size);
        sum_f = matrix_sum_3(mult_xt_Wf, mult_ht_Uf, matrix_bf, batch_size, hidden_size);
        f_t = matrix_sigmoid(sum_f, batch_size, hidden_size);

        // compute g_t
        mult_xt_Wc = matrix_mult(x_t, batch_size, input_size, W_c, input_size, hidden_size);
        mult_ht_Uc = matrix_mult(h_t, batch_size, hidden_size, U_c, hidden_size, hidden_size);
        sum_c = matrix_sum_3(mult_xt_Wc, mult_ht_Uc, matrix_bc, batch_size, hidden_size);
        g_t = matrix_tanh(sum_c, batch_size, hidden_size);

        // compute o_t
        mult_xt_Wo = matrix_mult(x_t, batch_size, input_size, W_o, input_size, hidden_size);
        mult_ht_Uo = matrix_mult(h_t, batch_size, hidden_size, U_o, hidden_size, hidden_size);
        sum_o = matrix_sum_3(mult_xt_Wo, mult_ht_Uo, matrix_bo, batch_size, hidden_size);
        o_t = matrix_sigmoid(sum_o, batch_size, hidden_size);

        // compute c_t
        mult_ft_ct = matrix_product(f_t, c_t, batch_size, hidden_size);
        mult_it_gt = matrix_product(i_t, g_t, batch_size, hidden_size);
        c_t = matrix_sum_2(mult_ft_ct, mult_it_gt, batch_size, hidden_size);

        // compute h_t
        tanh_ct = matrix_tanh(c_t, batch_size, hidden_size);
        h_t = matrix_product(o_t, tanh_ct, batch_size, hidden_size);
    }

    return h_t;
}

double** vectorized_forward_lstm(double x[batch_size][sequence_size][input_size],
double** W, double** U, double** matrix_bias, double** c_t_param, double** h_t_param,
double initial_weight){
    int i, j, k, m, n; 

    // x_t is one batch of our input x in each iteration
    double** x_t; // shape: (batch_size, input_size)
    x_t = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        x_t[i] = malloc(sizeof(double*) * input_size);
    }
    
    // used to compute gates matrix
    // we compute i_t, f_t, g_t, o_t using this matrix
    // all have shape: (batch_size, 4*hidden_size)
    double** mult_xt_W; 
    double** mult_ht_U; 

    double** gates; 
    gates = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) {
        gates[i] = malloc(sizeof(double*) * 4*hidden_size);
    }


    // used to compute i_t
    // shape: (batch_size, hidden_size)
    double** sliced_gates_i; 
    sliced_gates_i = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        sliced_gates_i[i] = malloc(sizeof(double*) * hidden_size);
    }

    double** i_t; 


    // used to compute f_t
    // shape: (batch_size, hidden_size)
    double** sliced_gates_f; 
    sliced_gates_f = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        sliced_gates_f[i] = malloc(sizeof(double*) * hidden_size);
    }

    double** f_t;


    // used to compute g_t
    // shape: (batch_size, hidden_size)
    double** sliced_gates_g; 
    sliced_gates_g = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        sliced_gates_g[i] = malloc(sizeof(double*) * hidden_size);
    }

    double** g_t;


    // used to compute o_t
    // shape: (batch_size, hidden_size)
    double** sliced_gates_o; 
    sliced_gates_o = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        sliced_gates_o[i] = malloc(sizeof(double*) * hidden_size);
    }

    double** o_t; 


    // used to compute c_t
    // shape: (batch_size, hidden_size)
    double** mult_ft_ct; 
    double** mult_it_gt; 

    // used to compute h_t
    // shape: (batch_size, hidden_size)
    double** tanh_ct; 

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


    // create matrix c_t and initialize it with the values from the input of function
    double** c_t;
    c_t = malloc(sizeof(double*) * batch_size);
    for(i = 0; i < batch_size; i++) { 
        c_t[i] = malloc(sizeof(double*) * hidden_size);
    }

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            c_t[i][j] = c_t_param[i][j];
        }
    }


    // iterate on each sequence and compute h_t and c_t
    for (j = 0; j < sequence_size; j++){
        for (i = 0; i < batch_size; i++){
            for (k = 0; k < input_size; k++){
                x_t[i][k] = x[i][j][k];
            }
        }

        // compute gates
        mult_xt_W = matrix_mult(x_t, batch_size, input_size, W, input_size, 4*hidden_size);
        mult_ht_U = matrix_mult(h_t, batch_size, hidden_size, U, hidden_size, 4*hidden_size);
        gates = matrix_sum_3(mult_xt_W, mult_ht_U, matrix_bias, batch_size, 4*hidden_size);

        // compute i_t
        for (m = 0; m < batch_size; m++){
            for (n = 0; n < hidden_size; n++){
                sliced_gates_i[m][n] = gates[m][n];
            }
        }

        i_t = matrix_sigmoid(sliced_gates_i, batch_size, hidden_size);


        // compute f_t
        for (m = 0; m < batch_size; m++){
            for (n = 0; n < hidden_size; n++){
                sliced_gates_f[m][n] = gates[m][n];
            }
        }

        f_t = matrix_sigmoid(sliced_gates_f, batch_size, hidden_size);


        // compute g_t
        for (m = 0; m < batch_size; m++){
            for (n = 0; n < hidden_size; n++){
                sliced_gates_g[m][n] = gates[m][n];
            }
        }

        g_t = matrix_tanh(sliced_gates_g, batch_size, hidden_size);


        // compute o_t
        for (m = 0; m < batch_size; m++){
            for (n = 0; n < hidden_size; n++){
                sliced_gates_o[m][n] = gates[m][n];
            }
        }

        o_t = matrix_sigmoid(sliced_gates_o, batch_size, hidden_size);
        

        // compute c_t
        mult_ft_ct = matrix_product(f_t, c_t, batch_size, hidden_size);
        mult_it_gt = matrix_product(i_t, g_t, batch_size, hidden_size);
        c_t = matrix_sum_2(mult_ft_ct, mult_it_gt, batch_size, hidden_size);

        // compute h_t
        tanh_ct = matrix_tanh(c_t, batch_size, hidden_size);
        h_t = matrix_product(o_t, tanh_ct, batch_size, hidden_size);
    }

    return h_t;
}

double** vectorized_backward_lstm(double x[batch_size][sequence_size][input_size],
double** W, double** U, double** matrix_bias, double** c_t_param, double** h_t_param,
double initial_weight){
    int i, j, k, m, n; 

    // x_t is one batch of our input x in each iteration
    double** x_t; // shape: (batch_size, input_size)
    x_t = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        x_t[i] = malloc(sizeof(double*) * input_size);
    }
    
    // used to compute gates matrix
    // we compute i_t, f_t, g_t, o_t using this matrix
    // all have shape: (batch_size, 4*hidden_size)
    double** mult_xt_W; 
    double** mult_ht_U; 

    double** gates; 
    gates = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) {
        gates[i] = malloc(sizeof(double*) * 4*hidden_size);
    }


    // used to compute i_t
    // shape: (batch_size, hidden_size)
    double** sliced_gates_i; 
    sliced_gates_i = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        sliced_gates_i[i] = malloc(sizeof(double*) * hidden_size);
    }

    double** i_t; 


    // used to compute f_t
    // shape: (batch_size, hidden_size)
    double** sliced_gates_f; 
    sliced_gates_f = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        sliced_gates_f[i] = malloc(sizeof(double*) * hidden_size);
    }

    double** f_t;


    // used to compute g_t
    // shape: (batch_size, hidden_size)
    double** sliced_gates_g; 
    sliced_gates_g = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        sliced_gates_g[i] = malloc(sizeof(double*) * hidden_size);
    }

    double** g_t;


    // used to compute o_t
    // shape: (batch_size, hidden_size)
    double** sliced_gates_o; 
    sliced_gates_o = malloc(sizeof(double*) * batch_size); 
    for(i = 0; i < batch_size; i++) { 
        sliced_gates_o[i] = malloc(sizeof(double*) * hidden_size);
    }

    double** o_t; 


    // used to compute c_t
    // shape: (batch_size, hidden_size)
    double** mult_ft_ct; 
    double** mult_it_gt; 

    // used to compute h_t
    // shape: (batch_size, hidden_size)
    double** tanh_ct; 

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


    // create matrix c_t and initialize it with the values from the input of function
    double** c_t;
    c_t = malloc(sizeof(double*) * batch_size);
    for(i = 0; i < batch_size; i++) { 
        c_t[i] = malloc(sizeof(double*) * hidden_size);
    }

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            c_t[i][j] = c_t_param[i][j];
        }
    }


    // iterate on each sequence and compute h_t and c_t
    for (j = sequence_size-1; j >= 0; j--){
        for (i = 0; i < batch_size; i++){
            for (k = 0; k < input_size; k++){
                x_t[i][k] = x[i][j][k];
            }
        }

        // compute gates
        mult_xt_W = matrix_mult(x_t, batch_size, input_size, W, input_size, 4*hidden_size);
        mult_ht_U = matrix_mult(h_t, batch_size, hidden_size, U, hidden_size, 4*hidden_size);
        gates = matrix_sum_3(mult_xt_W, mult_ht_U, matrix_bias, batch_size, 4*hidden_size);

        // compute i_t
        for (m = 0; m < batch_size; m++){
            for (n = 0; n < hidden_size; n++){
                sliced_gates_i[m][n] = gates[m][n];
            }
        }

        i_t = matrix_sigmoid(sliced_gates_i, batch_size, hidden_size);


        // compute f_t
        for (m = 0; m < batch_size; m++){
            for (n = 0; n < hidden_size; n++){
                sliced_gates_f[m][n] = gates[m][n];
            }
        }

        f_t = matrix_sigmoid(sliced_gates_f, batch_size, hidden_size);


        // compute g_t
        for (m = 0; m < batch_size; m++){
            for (n = 0; n < hidden_size; n++){
                sliced_gates_g[m][n] = gates[m][n];
            }
        }

        g_t = matrix_tanh(sliced_gates_g, batch_size, hidden_size);


        // compute o_t
        for (m = 0; m < batch_size; m++){
            for (n = 0; n < hidden_size; n++){
                sliced_gates_o[m][n] = gates[m][n];
            }
        }

        o_t = matrix_sigmoid(sliced_gates_o, batch_size, hidden_size);
        

        // compute c_t
        mult_ft_ct = matrix_product(f_t, c_t, batch_size, hidden_size);
        mult_it_gt = matrix_product(i_t, g_t, batch_size, hidden_size);
        c_t = matrix_sum_2(mult_ft_ct, mult_it_gt, batch_size, hidden_size);

        // compute h_t
        tanh_ct = matrix_tanh(c_t, batch_size, hidden_size);
        h_t = matrix_product(o_t, tanh_ct, batch_size, hidden_size);
    }

    return h_t;
}

double** bi_lstm(double** h_t_forward, double** h_t_backward, char merge_mode[]){
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