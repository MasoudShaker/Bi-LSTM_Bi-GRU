#include "gru_funcs.h"


int main(){
    double stdv = 1.0 / sqrt(hidden_size);
    // initial weight of W, U, b matrices
    double initial_weight = stdv;

    int i, j, k;

    // define and initialize W_sth, U_sth, b_sth matrices
    // all W_sth matrices have shape: (input_size, hidden_size)
    // all U_sth matrices have shape: (hidden_size, hidden_size)
    // all b vectors have size: hiddden_size

    // used to compute z_t
    double** W_z;
    W_z = malloc(sizeof(double*) * input_size);
    for(i = 0; i < input_size; i++) { 
        W_z[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(W_z, input_size, hidden_size, initial_weight);


    double** U_z; 
    U_z = malloc(sizeof(double*) * hidden_size);
    for(i = 0; i < hidden_size; i++) { 
        U_z[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(U_z, hidden_size, hidden_size, initial_weight);
    

    double* b_z;
    b_z = malloc(sizeof(double) * hidden_size);

    initialize_vector(b_z, hidden_size, initial_weight);


    // used to compute r_t
    double** W_r; 
    W_r = malloc(sizeof(double*) * input_size);
    for(i = 0; i < input_size; i++) { 
        W_r[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(W_r, input_size, hidden_size, initial_weight);


    double** U_r; 
    U_r = malloc(sizeof(double*) * hidden_size);
    for(i = 0; i < hidden_size; i++) { 
        U_r[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(U_r, hidden_size, hidden_size, initial_weight);


    double* b_r;
    b_r = malloc(sizeof(double) * hidden_size);

    initialize_vector(b_r, hidden_size, initial_weight);


    // used to compute h_tilde
    double** W_h; 
    W_h = malloc(sizeof(double*) * input_size);
    for(i = 0; i < input_size; i++) { 
        W_h[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(W_h, input_size, hidden_size, initial_weight);


    double** U_h; 
    U_h = malloc(sizeof(double*) * hidden_size);
    for(i = 0; i < hidden_size; i++) { 
        U_h[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(U_h, hidden_size, hidden_size, initial_weight);


    double* b_h;
    b_h = malloc(sizeof(double) * hidden_size);

    initialize_vector(b_h, hidden_size, initial_weight);


    // define broadcasted version of b_sth vectors
    double** matrix_bz; 
    double** matrix_br; 
    double** matrix_bh; 

    // broadcast b_sth vectors to matrices
    // needed for our computings
    matrix_bz = broadcast_vector_to_matrix(b_z, batch_size, hidden_size);
    matrix_br = broadcast_vector_to_matrix(b_r, batch_size, hidden_size);
    matrix_bh = broadcast_vector_to_matrix(b_h, batch_size, hidden_size);

    // define and initialize hidden state
    // shape: (batch_size, hidden_size)
    double** h_t;
    h_t = malloc(sizeof(double*) * batch_size);
    for(i = 0; i < batch_size; i++) { 
        h_t[i] = malloc(sizeof(double*) * hidden_size);
    }

    initialize_matrix(h_t, batch_size, hidden_size, 0);


    // define and initialize the input sequnce of the "forward_gru" function
    double x[batch_size][sequence_size][input_size];

    // for (i = 0; i < batch_size; i++){
    //     for (j = 0; j < sequence_size; j++){
    //         for (k = 0; k < input_size; k++){
    //             x[i][j][k] = initial_weight;
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


    // outputs of the forward and backward gru
    double** forward_h_t_output;
    double** backward_h_t_output;

    // output of bi_lstm model
    double** bi_gru_h_t_output;

    // call the forward and backward gru functions and get forward h_t and backward h_t
    forward_h_t_output = forward_gru(x, W_z, U_z, matrix_bz, W_r, U_r,
    matrix_br, W_h, U_h, matrix_bh, h_t, initial_weight);

    backward_h_t_output = backward_gru(x, W_z, U_z, matrix_bz, W_r, U_r,
    matrix_br, W_h, U_h, matrix_bh, h_t, initial_weight);


    // call bi_gru model and get the final h_t 
    // four values accepted for merge_mode: "sum", "mult", "avg", "concat"
    char merge_mode[] = "concat";
    bi_gru_h_t_output = bi_gru(forward_h_t_output, backward_h_t_output, merge_mode);


    printf("********************************************************\n");
    printf("\nforward gru output: \n\n");

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            printf("%f\t", forward_h_t_output[i][j]);
        }
        printf("\n");
    }

    printf("\n");

    
    printf("********************************************************\n");
    printf("\nbackward gru output: \n\n");

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < hidden_size; j++){
            printf("%f\t", backward_h_t_output[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    

    printf("********************************************************\n");
    printf("\nbi gru output: \n\n");

    for (i = 0; i < batch_size; i++){
        for (j = 0; j < 2*hidden_size; j++){
            printf("%f\t", bi_gru_h_t_output[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("********************************************************");


    return 0;
}