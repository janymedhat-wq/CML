
# Mini Neural Network & Linear Regression Library in C

This project is a simple, educational machine learning library written in C.  
It demonstrates the basics of linear regression and neural networks, including vector/matrix operations, activation functions, and loss calculation.

## Features

- **Linear Regression:** Train and evaluate a simple linear regression model.
- **Vector & Matrix Operations:** Addition, subtraction, multiplication, dot product, transpose, and more.
- **Dense Neural Network Layer:** Fully connected layer with customizable activation functions.
- **Activation Functions:** ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax.
- **Forward Pass:** Compute neural network outputs for given inputs.
- **Cross-Entropy Loss:** For classification tasks.
- **Memory Management:** Safe allocation and deallocation for all structures.
- **Demonstration:** Example usage in `main()` for both regression and neural network classification.

## How to Build & Run

1. **Compile:**
    ```sh
    gcc -o nerun nerun.c -lm
    ```

2. **Run:**
    ```sh
    ./nerun
    ```

## Example Output

```
--- Linear Regression Demonstration ---
Model trained successfully!
Slope (m): 1.0400
Y-intercept (b): 0.8600
Mean Squared Error (MSE): 0.2120

--- Dense Layer with Softmax and Cross-Entropy Loss Demonstration ---
Input Vector:
[0.8000, 1.2000, 0.5000]
Created Dense Layer with 3 inputs and 3 outputs for classification.
Predicted Output (after Softmax activation):
[0.3123, 0.3456, 0.3421]
Actual Label:
[0.0000, 1.0000, 0.0000]
Cross-Entropy Loss: 1.0623
```

## File Structure

- `nerun.c` — Main source file with all code and demo.

## License

MIT License 
