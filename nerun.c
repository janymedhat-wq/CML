#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct
{
    double m;
    double b;
} LinearRegressionModel;

typedef struct
{
    double x;
    double y;
} DataPoint;

typedef struct
{
    int size;
    double *data;
} Vector;

typedef struct
{
    int rows;
    int cols;
    double *data;
} Matrix;

typedef struct
{
    Matrix *weights;
    Vector *biases;
    void (*activation)(Vector *);
} DenseLayer;

Vector *new_vector(int size)
{
    Vector *v = (Vector *)malloc(sizeof(Vector));
    if (!v)
        return NULL;
    v->size = size;
    v->data = (double *)calloc(size, sizeof(double));
    if (!v->data)
    {
        free(v);
        return NULL;
    }
    return v;
}

void free_vector(Vector *v)
{
    if (v)
    {
        if (v->data)
        {
            free(v->data);
        }
        free(v);
    }
}

Matrix *new_matrix(int rows, int cols)
{
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    if (!m)
        return NULL;
    m->rows = rows;
    m->cols = cols;
    m->data = (double *)calloc(rows * cols, sizeof(double));
    if (!m->data)
    {
        free(m);
        return NULL;
    }
    return m;
}

void free_matrix(Matrix *m)
{
    if (m)
    {
        if (m->data)
        {
            free(m->data);
        }
        free(m);
    }
}

DenseLayer *new_dense_layer(int input_size, int output_size, void (*activation_func)(Vector *))
{
    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    if (!layer)
        return NULL;

    layer->weights = new_matrix(output_size, input_size);
    layer->biases = new_vector(output_size);
    layer->activation = activation_func;

    random_matrix_uniform(layer->weights);

    return layer;
}

void free_dense_layer(DenseLayer *layer)
{
    if (layer)
    {
        if (layer->weights)
        {
            free_matrix(layer->weights);
        }
        if (layer->biases)
        {
            free_vector(layer->biases);
        }
        free(layer);
    }
}

double calculate_mean(double values[], int count)
{
    double sum = 0.0;
    for (int i = 0; i < count; i++)
    {
        sum += values[i];
    }
    return sum / count;
}

void train_model(DataPoint data[], int count, LinearRegressionModel *model)
{
    if (count < 2)
    {
        printf("Error: Not enough data points to train the model.\n");
        model->m = 0.0;
        model->b = 0.0;
        return;
    }

    double x_values[count];
    double y_values[count];
    for (int i = 0; i < count; i++)
    {
        x_values[i] = data[i].x;
        y_values[i] = data[i].y;
    }

    double mean_x = calculate_mean(x_values, count);
    double mean_y = calculate_mean(y_values, count);

    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < count; i++)
    {
        numerator += (x_values[i] - mean_x) * (y_values[i] - mean_y);
        denominator += (x_values[i] - mean_x) * (x_values[i] - mean_x);
    }

    if (denominator == 0)
    {
        printf("Error: All x-values are identical, cannot compute slope.\n");
        model->m = 0.0;
        model->b = mean_y;
        return;
    }

    model->m = numerator / denominator;
    model->b = mean_y - model->m * mean_x;
}double predict(LinearRegressionModel *model, double x)
{
    return model->m * x + model->b;
}double calculate_mse(LinearRegressionModel *model, DataPoint data[], int count)
{
    double total_error = 0.0;
    for (int i = 0; i < count; i++)
    {
        double predicted_y = predict(model, data[i].x);
        double actual_y = data[i].y;
        total_error += pow(actual_y - predicted_y, 2);
    }
    return total_error / count;
}

void vector_add(const Vector *v1, const Vector *v2, Vector *result)
{
    if (v1->size != v2->size || v1->size != result->size)
    {
        printf("Error: Vectors must have the same size for addition.\n");
        return;
    }
    for (int i = 0; i < v1->size; i++)
    {
        result->data[i] = v1->data[i] + v2->data[i];
    }
}

void vector_subtract(const Vector *v1, const Vector *v2, Vector *result)
{
    if (v1->size != v2->size || v1->size != result->size)
    {
        printf("Error: Vectors must have the same size for subtraction.\n");
        return;
    }
    for (int i = 0; i < v1->size; i++)
    {
        result->data[i] = v1->data[i] - v2->data[i];
    }
}

void vector_multiply(const Vector *v1, const Vector *v2, Vector *result)
{
    if (v1->size != v2->size || v1->size != result->size)
    {
        printf("Error: Vectors must have the same size for multiplication.\n");
        return;
    }
    for (int i = 0; i < v1->size; i++)
    {
        result->data[i] = v1->data[i] * v2->data[i];
    }
}

double vector_dot(const Vector *v1, const Vector *v2)
{
    if (v1->size != v2->size)
    {
        printf("Error: Vectors must have the same size for dot product.\n");
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < v1->size; i++)
    {
        sum += v1->data[i] * v2->data[i];
    }
    return sum;
}

void vector_exp(Vector *v)
{
    for (int i = 0; i < v->size; i++)
    {
        v->data[i] = exp(v->data[i]);
    }
}

void matrix_add(const Matrix *m1, const Matrix *m2, Matrix *result)
{
    if (m1->rows != m2->rows || m1->cols != m2->cols)
    {
        printf("Error: Matrices must have the same dimensions for addition.\n");
        return;
    }
    for (int i = 0; i < m1->rows * m1->cols; i++) {
        result->data[i] = m1->data[i] + m2->data[i];
    }
}void matrix_subtract(const Matrix *m1, const Matrix *m2, Matrix *result)
{
    if (m1->rows != m2->rows || m1->cols != m2->cols)
    {
        printf("Error: Matrices must have the same dimensions for subtraction.\n");
        return printf("Error"); 
        
    }
    for (int i = 0; i < m1->rows * m1->cols; i++)
    {
        result->data[i] = m1->data[i] - m2->data[i];
    }
}

void matrix_multiply_elementwise(const Matrix *m1, const Matrix *m2, Matrix *result)
{
    if (m1->rows != m2->rows || m1->cols != m2->cols)
    {
        printf("Error: Matrices must have the same dimensions for element-wise multiplication.\n");
        return;
    }
    for (int i = 0; i < m1->rows * m1->cols; i++)
    {
        result->data[i] = m1->data[i] * m2->data[i];
    }
}

void matrix_multiply(const Matrix *m1, const Matrix *m2, Matrix *result)
{
    if (m1->cols != m2->rows)
    {
        printf("Error: The number of columns in the first matrix must equal the number of rows in the second.\n");
        return;
    }
    if (result->rows != m1->rows || result->cols != m2->cols)
    {
        printf("Error: Result matrix has incorrect dimensions for multiplication.\n");
        return;
    }

    for (int i = 0; i < m1->rows; i++)
    {
        for (int j = 0; j < m2->cols; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < m1->cols; k++)
            {
                sum += m1->data[i * m1->cols + k] * m2->data[k * m2->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
}

void matrix_vector_multiply(const Matrix *m, const Vector *v, Vector *result)
{
    if (m->cols != v->size)
    {
        printf("Error: Matrix columns must equal vector size for multiplication.\n");
        return;
    }
    if (result->size != m->rows)
    {
        printf("Error: Result vector size must equal matrix rows.\n");
        return;
    }

    for (int i = 0; i < m->rows; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < m->cols; j++)
        {
            sum += m->data[i * m->cols + j] * v->data[j];
        }
        result->data[i] = sum;
    }
}

void matrix_transpose(const Matrix *original, Matrix *transposed)
{
    if (original->rows != transposed->cols || original->cols != transposed->rows)
    {
        printf("Error: Transposed matrix dimensions are incorrect.\n");
        return;
    }
    for (int i = 0; i < original->rows; i++)
    {
        for (int j = 0; j < original->cols; j++)
        {
            transposed->data[j * transposed->cols + i] = original->data[i * original->cols + j];
        }
    }
}

void random_vector_uniform(Vector *v)
{
    for (int i = 0; i < v->size; i++)
    {
        v->data[i] = (double)rand() / RAND_MAX;
    }
}

void random_matrix_uniform(Matrix *m)
{
    for (int i = 0; i < m->rows * m->cols; i++)
    {
        m->data[i] = (double)rand() / RAND_MAX;
    }
}

void relu_vector(Vector *v)
{
    for (int i = 0; i < v->size; i++)
    {
        v->data[i] = fmax(0.0, v->data[i]);
    }
}

void sigmoid_vector(Vector *v)
{
    for (int i = 0; i < v->size; i++)
    {
        v->data[i] = 1.0 / (1.0 + exp(-v->data[i]));
    }
}

void tanh_vector(Vector *v)
{
    for (int i = 0; i < v->size; i++)
    {
        v->data[i] = tanh(v->data[i]);
    }
}

void leaky_relu_vector(Vector *v, double alpha)
{
    for (int i = 0; i < v->size; i++)
    {
        v->data[i] = (v->data[i] > 0) ? v->data[i] : alpha * v->data[i];
    }
}

void softmax_vector(Vector *v)
{
    vector_exp(v);
    double sum = 0.0;
    for (int i = 0; i < v->size; i++)
    {
        sum += v->data[i];
    }
    for (int i = 0; i < v->size; i++)
    {
        v->data[i] /= sum;
    }
}

Vector *forward_pass(DenseLayer *layer, const Vector *input)
{
    Vector *output = new_vector(layer->biases->size);
    if (!output)
        return NULL;

    matrix_vector_multiply(layer->weights, input, output);
    vector_add(output, layer->biases, output);

    if (layer->activation)
    {
        layer->activation(output);
    }

    return output;
}

double cross_entropy_loss(const Vector *predicted, const Vector *actual)
{
    if (predicted->size != actual->size)
    {
        printf("Error: Predicted and actual vectors must have the same size.\n");
        return 0.0;
    }
    double loss = 0.0;
    for (int i = 0; i < predicted->size; i++)
    {
        loss -= actual->data[i] * log(predicted->data[i] + 1e-9);
    }
    return loss;
}

void print_vector(const Vector *v)
{
    printf("[");
    for (int i = 0; i < v->size; i++)
    {
        printf("%.4f", v->data[i]);
        if (i < v->size - 1)
        {
            printf(", ");
        }
    }
    printf("]\n");
}

void print_matrix(const Matrix *m)
{
    printf("[\n");
    for (int i = 0; i < m->rows; i++)
    {
        printf("  [");
        for (int j = 0; j < m->cols; j++)
        {
            printf("%.4f", m->data[i * m->cols + j]);
            if (j < m->cols - 1)
            {
                printf(", ");
            }
        }
        printf("]");
        if (i < m->rows - 1)
        {
            printf(",\n");
        }
    }
    printf("\n]\n");
}

int main()
{
    srand(time(NULL));

    printf("--- Linear Regression Demonstration ---\n");
    DataPoint dataset[] = {
        {1.0, 2.0},
        {2.0, 3.5},
        {3.0, 4.0},
        {4.0, 5.5},
        {5.0, 6.0}};
    int data_size = sizeof(dataset) / sizeof(dataset[0]);

    LinearRegressionModel my_model;
    train_model(dataset, data_size, &my_model);
    printf("Model trained successfully!\n");
    printf("Slope (m): %.4f\n", my_model.m);
    printf("Y-intercept (b): %.4f\n", my_model.b);
    double mse = calculate_mse(&my_model, dataset, data_size);
    printf("Mean Squared Error (MSE): %.4f\n\n", mse);

    printf("--- Dense Layer with Softmax and Cross-Entropy Loss Demonstration ---\n");

    Vector *input_vector = new_vector(3);
    input_vector->data[0] = 0.8;
    input_vector->data[1] = 1.2;
    input_vector->data[2] = 0.5;

    printf("Input Vector:\n");
    print_vector(input_vector);

    DenseLayer *dense_layer = new_dense_layer(3, 3, softmax_vector);
    printf("\nCreated Dense Layer with 3 inputs and 3 outputs for classification.\n");

    Vector *predicted_output = forward_pass(dense_layer, input_vector);
    printf("Predicted Output (after Softmax activation):\n");
    print_vector(predicted_output);

    Vector *actual_label = new_vector(3);
    actual_label->data[0] = 0.0;
    actual_label->data[1] = 1.0;
    actual_label->data[2] = 0.0;
    printf("\nActual Label:\n");
    print_vector(actual_label);

    double loss = cross_entropy_loss(predicted_output, actual_label);
    printf("Cross-Entropy Loss: %.4f\n", loss);

    free_vector(input_vector);
    free_vector(predicted_output);
    free_vector(actual_label);
    free_dense_layer(dense_layer);

    return 0;
}
