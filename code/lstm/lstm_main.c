#include "lstm_funcs.h"


int main(){
    double stdv = 1.0 / sqrt(hidden_size);
    // initial weight of W, U, b matrices
    double initial_weight = stdv;

    int i, j, k;

    // ********** Definitions and Initializations needed for forward_lstm and backward lstm start **********

    // define and initialize W_sth, U_sth, b_sth matrices
    // all W_sth matrices have shape: (input_size, hidden_size)
    // all U_sth matrices have shape: (hidden_size, hidden_size)
    // all b_sth vectors have size: hiddden_size

    // used to compute i_t (input gate)
    double** W_i; 
    W_i = malloc(sizeof(double*) * input_size);
    for(i = 0; i < input_size; i++) { 
        W_i[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(W_i, input_size, hidden_size, initial_weight);


    double** U_i; 
    U_i = malloc(sizeof(double*) * hidden_size);
    for(i = 0; i < hidden_size; i++) { 
        U_i[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(U_i, hidden_size, hidden_size, initial_weight);


    double* b_i;
    b_i = malloc(sizeof(double) * hidden_size);

    initialize_vector(b_i, hidden_size, initial_weight);


    // used to compute f_t (forget gate)
    double** W_f; 
    W_f = malloc(sizeof(double*) * input_size);
    for(i = 0; i < input_size; i++) { 
        W_f[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(W_f, input_size, hidden_size, initial_weight);


    double** U_f; 
    U_f = malloc(sizeof(double*) * hidden_size);
    for(i = 0; i < hidden_size; i++) { 
        U_f[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(U_f, hidden_size, hidden_size, initial_weight);


    double* b_f;
    b_f = malloc(sizeof(double) * hidden_size);

    initialize_vector(b_f, hidden_size, initial_weight);


    // used to compute c_t (cell state)
    double** W_c; 
    W_c = malloc(sizeof(double*) * input_size);
    for(i = 0; i < input_size; i++) { 
        W_c[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(W_c, input_size, hidden_size, initial_weight);


    double** U_c; 
    U_c = malloc(sizeof(double*) * hidden_size);
    for(i = 0; i < hidden_size; i++) { 
        U_c[i] = malloc(sizeof(double*) * hidden_size);
    }
    
    initialize_matrix(U_c, hidden_size, hidden_size, initial_weight);


    double* b_c;
    b_c = malloc(sizeof(double) * hidden_size);

    initialize_vector(b_c, hidden_size, initial_weight);


    // used to compute o_t (output gate)
    double** W_o; 
    W_o = malloc(sizeof(double*) * input_size);
    for(i = 0; i < input_size; i++) { 
        W_o[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(W_o, input_size, hidden_size, initial_weight);


    double** U_o; 
    U_o = malloc(sizeof(double*) * hidden_size);
    for(i = 0; i < hidden_size; i++) { 
        U_o[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(U_o, hidden_size, hidden_size, initial_weight);


    double* b_o;
    b_o = malloc(sizeof(double) * hidden_size);

    initialize_vector(b_o, hidden_size, initial_weight);


    // define broadcasted version of b_sth vectorz
    double** matrix_bi;
    double** matrix_bf;
    double** matrix_bc;
    double** matrix_bo;

    // broadcast b_sth vectors to matrices
    // needed for our computings
    matrix_bi = broadcast_vector_to_matrix(b_i, batch_size, hidden_size);
    matrix_bf = broadcast_vector_to_matrix(b_f, batch_size, hidden_size);
    matrix_bc = broadcast_vector_to_matrix(b_c, batch_size, hidden_size);
    matrix_bo = broadcast_vector_to_matrix(b_o, batch_size, hidden_size);

    // define and initialize cell state
    // shape: (batch_size, hidden_size)
    double** c_t;
    c_t = malloc(sizeof(double*) * batch_size);
    for(i = 0; i < batch_size; i++) { 
        c_t[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(c_t, batch_size, hidden_size, 0);


    // define and initialize hidden state
    // shape: (batch_size, hidden_size)
    double** h_t;
    h_t = malloc(sizeof(double*) * batch_size);
    for(i = 0; i < batch_size; i++) { 
        h_t[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(h_t, batch_size, hidden_size, 0);

    // ********** Definitions and Initializations needed for forward_lstm and backward lstm end **********

    // ********** Definitions and Initializations needed for vectorized_forward_lstm and vectorized_backward lstm start **********
    
    // define and initialize W, U, bias Matrices
    // W shape: (input_size, 4*hidden_size)
    // U shape: (hidden_size, 4*hidden_size)
    // bias size: 4*hidden_size
    double** W;
    W = malloc(sizeof(double*) * input_size);
    for(i = 0; i < input_size; i++) { 
        W[i] = malloc(sizeof(double*) * 4*hidden_size);
    }

    initialize_matrix(W, input_size, 4*hidden_size, initial_weight);


    double** U; 
    U = malloc(sizeof(double*) * hidden_size);
    for(i = 0; i < hidden_size; i++) { 
        U[i] = malloc(sizeof(double*) * 4*hidden_size);
    }

    initialize_matrix(U, hidden_size, 4*hidden_size, initial_weight);


    double* bias; 
    bias = malloc(sizeof(double) * 4*hidden_size);

    initialize_vector(bias, 4*hidden_size, initial_weight);

    double** matrix_bias;
    // broadcast bias vector to matrix
    // needed for our computing 
    matrix_bias = broadcast_vector_to_matrix(bias, batch_size, 4*hidden_size);

    // ********** Definitions and Initializations needed for vectorized_forward_lstm and vectorized_backward lstm end **********

    // outputs of the forward and backward lstms
    double** forward_h_t_output;
    double** backward_h_t_output;

    double** vectorized_forward_h_t_output;
    double** vectorized_backward_h_t_output;


    // output of bi_lstm model
    double** bi_lstm_h_t_output;

    // define and initialize the input of the lstm model
    double x[batch_size][sequence_size][input_size];

    // for (i = 0; i < batch_size; i++){
    //     for (j = 0; j < sequence_size; j++){
    //         for (k = 0; k < input_size; k++){
    //             x[i][j][k] = 1;
    //         }
    //     }
    // }


    x[0][0][0] = 0.5;
    x[0][0][1] = 0.6;
    x[0][1][0] = 0.7;
    x[0][1][1] = 0.8;

    x[1][0][0] = 0.9;
    x[1][0][1] = 1.0;
    x[1][1][0] = 1.1;
    x[1][1][1] = 1.2;


    // call the forward and backward lstm functions and get forward h_t and backward h_t
    forward_h_t_output = forward_lstm(x, W_i, U_i, matrix_bi, W_f, U_f,matrix_bf,
    W_c, U_c, matrix_bc, W_o, U_o, matrix_bo, c_t, h_t, initial_weight);

    vectorized_forward_h_t_output = vectorized_forward_lstm(x, W, U, matrix_bias, c_t, h_t, initial_weight);

    backward_h_t_output = backward_lstm(x, W_i, U_i, matrix_bi, W_f, U_f,matrix_bf,
    W_c, U_c, matrix_bc, W_o, U_o, matrix_bo, c_t, h_t, initial_weight);

    vectorized_backward_h_t_output = vectorized_backward_lstm(x, W, U, matrix_bias, c_t, h_t, initial_weight);


    // call bi_lstm model and get the final h_t 
    // four values accepted for merge_mode: "sum", "mult", "avg", "concat"
    char merge_mode[] = "concat";
    bi_lstm_h_t_output = bi_lstm(vectorized_forward_h_t_output, vectorized_backward_h_t_output, merge_mode);


    printf("********************************************************\n");
    printf("\nforward lstm output: \n\n");

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            printf("%f\t", forward_h_t_output[i][j]);
        }
        printf("\n");
    }

    printf("\n");


    printf("********************************************************\n");
    printf("\nbackward lstm output: \n\n");

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            printf("%f\t", backward_h_t_output[i][j]);
        }
        printf("\n");
    }

    printf("\n");


    printf("********************************************************\n");
    printf("\nbi lstm output: \n\n");

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < 2*hidden_size; j++){
            printf("%f\t", bi_lstm_h_t_output[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("********************************************************");


    return 0;
}