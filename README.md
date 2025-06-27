# MLP Neural Net. for Handwritten Digit Recognition :pushpin:
This project implements a **Multilayer Perceptron (MLP)** neural network designed to recognize handwritten digits using a dataset of image samples. 

:dart: The goal was to **build and train** a model that achieves high classification accuracy, assess its performance using confusion matrices, and test the predictions on a representative sample of digits **from 0 to 9**.

> :paperclip: *Additionally, documentation in **Russian and Slovak** will soon be available in the files `README_rus.md` and `README_slk.md`*

*Content of the documentation* :arrow_down:
## Content:
>* :repeat: [Description of Input and Output Data](#repeat-description-of-input-and-output-data)
>* :page_facing_up: [MLP Network Structure](#page_facing_up-mlp-network-structure)
>* :surfer: [Training Parameters](#surfer-training-parameters)
>   * [Termination Conditions](#termination-conditions)
>   * [Criterion Function](#criterion-function)
>* [Training Process and Contingency Matrix for the Best Network](#training-process-and-contingency-matrix-for-the-best-network)
>   * :chart_with_downwards_trend: [Training Process Progress Chart](#chart_with_downwards_trend-training-process-progress-chart)
>   * :1234: [Contingency Matrix (plotconfusion)](#1234-contingency-matrix-plotconfusion)
>* :clipboard: [Neural Network Testing](#clipboard-neural-network-testing)
>* :page_facing_up: [Training Process and Contingency Matrix for Different Neuron Counts](#page_facing_up-training-process-and-contingency-matrix-for-different-neuron-counts)
>   * [First Variant: 10 Neurons](#first-variant-10-neurons)
>   * [Second Variant: 200 Neurons](#second-variant-200-neurons)
>* :pill: [Testing with One Sample per Digit](#pill-testing-with-one-sample-per-digit)

### :repeat: Description of Input and Output Data
#### *INPUT DATA*
*The data is loaded from the variable 'XDataall', stored in the file 'datapiscisla_all.mat'.*
* The dataset contains **handwritten digits** represented as **28×28 grayscale images**, resulting in **784 pixels per character**.
* **Pixel values range from 0 to 255**, where higher values correspond to **darker pixels**.
>```matlab
>% Data loading
>data = load('datapiscisla_all.mat');
>max_pre_pixel = 255;                                                        % Maximum pixel value in the image before normalization
>
>% Data normalization
>data.XDataall = data.XDataall / max_pre_pixel; 
>```
* After normalization, pixel values are scaled to the range **⟨0, 1⟩**:
   * A value of **0** represents a **white pixel** (no ink).
   * A value of **1** represents a **black pixel** (full ink intensity).
#### *OUTPUT DATA*
*YDataall:*
* Each character (digit) is assigned to one of **10 categories** representing digits **0 through 9**.
* The **target values** are encoded in **one-hot format**, where each digit is represented as a **binary vector** of length 10.
   * For example, the digit **3** is represented as: `[0 0 0 1 0 0 0 0 0 0]`.
     
This encoding allows the neural network to perform **multi-class classification** effectively using a softmax output layer.

#### *DATA SPLITTING INTO TRAINING AND TESTING*
*The dataset was **randomly divided** into training and testing subsets using MATLAB's built-in 'dividerand' function:*
>```matlab
>% Set data division using 'dividerand'
>net.divideFcn = 'dividerand';                                               % Randomly split the dataset
>net.divideParam.trainRatio = 0.6;                                           % 60% of data for training
>net.divideParam.valRatio = 0;                                               % 0% for validation
>net.divideParam.testRatio = 0.4;                                            % 40% of data for testing
>```

>:arrow_left: [**Back to *CONTENT***](#content)

### :page_facing_up: MLP Network Structure
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/3979fe5c-6fc2-4b3a-ac03-d5b566607ea7" width="200"></td>
    <td>
      <b><i>Inputs:</i></b>
      <ul>
        <li><b>Number of neurons:</b> <b>784</b> — one for each pixel in the 28×28 grayscale input image.</li>
        <li><b>Neuron type:</b> Passive input neurons that simply receive data and pass it forward.</li>
      </ul>
      <b><i>Hidden Layer:</i></b>
      <ul>
        <li><b>Number of neurons:</b> 65 (defined as <code>hidden_neurons = 65</code>).</li>
        <li><b>Neuron type:</b> Nonlinear neurons using an activation function.</li>
          <ul>
            <li>In <code>patternnet</code>, the default activation is <b>log-sigmoid</b> (<code>logsig</code>), which maps inputs to a range between 0 and 1.</li>
          </ul>  
      </ul>
      <b><i>Output Layer:</i></b>
      <ul>
        <li><b>Number of neurons:</b> 10 — one for each digit class (0 through 9).</li>
        <li><b>Neuron type:</b> Neurons use the <b>softmax</b> activation function to convert outputs into probabilities.</li>
          <ul>
            <li>Each output is in the range ⟨<b>0, 1</b>⟩ and the sum of all outputs equals 1.</li>
          </ul>
      </ul>
    </td>
  </tr>
</table>

>:arrow_left: [**Back to *CONTENT***](#content)

### :surfer: Training Parameters
```matlab
% Set training parameters
net.trainParam.goal = 0.00001;                                              % Desired error goal
net.trainParam.epochs = 1000;                                               % Maximum number of training epochs
net.trainParam.max_fail = 12;                                               % Maximum validation failures

% Train the neural network
[net, tr] = train(net, data.XDataall, data.YDataall);
```

* 'data.XDataall': Input data representing **handwritten digits**, normalized to the range ⟨0, 1⟩. Each digit is divided into a 28×28 grid, totaling 784 pixels per character.
* 'data.YDataall': Output data containing **target values in one-hot encoding** format, where each digit (0–9) is represented as a binary vector of length 10.

>:arrow_left: [**Back to *CONTENT***](#content)

#### *Termination Conditions:*
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/501dfd28-9e59-4825-94bc-d357c304484d" width="200"></td>
    <td>
      <ul>
        <li><i><code>net.trainParam.goal = 0.00001;</code></i></li>
          <ul>
            <li><i>Error-based stopping:</i> Training stops when the network's error reaches <b>0.00001</b>.</li>
          </ul>
        <li><i><code>net.trainParam.epochs = 1000;</code></i></li>
          <ul>
            <li><i>Maximum epochs:</i> The network can perform up to <b>1000 training cycles (epochs)</b>.</li>
          </ul>
        <li><i><code>net.trainParam.max_fail = 12;</code></i></li>
          <ul>
            <li><i>Maximum validation failures:</i> Training is halted if the network fails to improve over <b>12 consecutive validations</b>.</li>
          </ul>
      </ul>
    </td>
  </tr>
</table>

>:arrow_left: [**Back to *CONTENT***](#content)

#### *Criterion Function:*
The training process uses the **Mean Squared Error (MSE)** as the loss function. This function evaluates how well the neural network performs by calculating the average squared difference between predicted and actual outputs.

In the code:
* Training is terminated when the **MSE reaches the predefined goal value of** '0.00001' ('net.trainParam.goal = 0.00001').

>:arrow_left: [**Back to *CONTENT***](#content)

### Training Process and Contingency Matrix for the Best Network
> :grey_exclamation: *The best-performing network was achieved with 65 hidden neurons.*

#### :chart_with_downwards_trend: *Training Process Progress Chart:*
![image](https://github.com/user-attachments/assets/ec19f89d-1af0-45e5-b31b-faf74bcdf69c)

#### Key Elements of the Chart:
:large_blue_circle: *Blue Line (Training Data Error):*
   * The curve clearly shows that the training error decreases significantly over time, indicating that the model is gradually learning from the input samples and optimizing its weights.
   * At the **120th epoch**, the network reaches its lowest error value of **9.6667e-06**, suggesting excellent accuracy on the training data and close alignment with performance goals.

:red_circle: *Red Line (Testing Data Error):*
   * Initially, the testing error drops similarly to the training curve, but eventually stabilizes, indicating that the model maintains **consistent performance on unseen test data**.
   * A stable testing error is crucial for ensuring **robustness** and suggests that **overfitting is not present** — the network doesn't just memorize training data but generalizes well to new inputs.

>:arrow_left: [**Back to *CONTENT***](#content)

#### :1234: *Contingency Matrix (plotconfusion):*
![image](https://github.com/user-attachments/assets/19746f88-c3b0-42dc-9dd9-0b180b6c53f1)

This **confusion matrix provides a detailed view of the classification accuracy** during the neural network testing phase. It serves as a key indicator of the model’s learning success by displaying both correct and incorrect predictions.

In this particular case:
* The **overall test accuracy is 95.4%**,
* And the **error rate is 4.6%**, which **meets the performance requirements** set for the project.

>:arrow_left: [**Back to *CONTENT***](#content)

### :clipboard: *Neural Network Testing:*
>🔎 To validate the network’s robustness, the training process was repeated **5 times** using **randomized data splits** each run. The results of these runs are summarized below:
* **Training Accuracy**
   * *Min:* 100.00%
   * *Mean:* 100.00%
   * *Max:* 100.00%
* **Testing Accuracy**
   * *Min:* 95.47%
   * *Mean:* 95.47%
   * *Max:* 95.47%

📊 **MATLAB Implementation:**
```matlab
% Compute accuracy on training data
train_outputs = net(data.XDataall(:, tr.trainInd));                             
train_accuracies = 1 - confusion(data.YDataall(:, tr.trainInd), train_outputs);
train_accuracies = train_accuracies * 100;

% Compute accuracy on testing data
test_outputs = net(data.XDataall(:, tr.testInd));                               
test_accuracies = 1 - confusion(data.YDataall(:, tr.testInd), test_outputs);
test_accuracies = test_accuracies * 100;
testError = 100 - test_accuracies;                                              

...

% Display min, mean, and max accuracies
fprintf('Training Accuracy: min = %.2f%%, mean = %.2f%%, max = %.2f%%\n', ...
    min(train_accuracies), mean(train_accuracies), max(train_accuracies));
fprintf('Testing Accuracy: min = %.2f%%, mean = %.2f%%, max = %.2f%%\n', ...
    min(test_accuracies), mean(test_accuracies), max(test_accuracies));
```

>:arrow_left: [**Back to *CONTENT***](#content)

### :page_facing_up: Training Process and Contingency Matrix for Different Neuron Counts
#### First Variant: 10 Neurons 
![image](https://github.com/user-attachments/assets/8f780066-46fa-4ba7-9392-e11ccf84ee13)

#### Key Elements of the Chart:
:large_blue_circle: *Blue Line (Training Data Error):*
   * The training error **decreases significantly at the beginning**, indicating the model is learning.
   * However, the curve eventually reaches a plateau where further error reduction stops.
   * By the end of training (epoch 485), the network achieves a **minimum error of 9.9983e-06**, but **only on the training data**, which may not reflect general performance.

:red_circle: *Red Line (Testing Data Error):*
   * The test error remains relatively **stable throughout training**, without a notable decrease.
   * This suggests that a network with only 10 neurons has **limited capacity to capture complex patterns** in the data.
   * While the stable error implies **no overfitting**, the relatively high test error indicates **suboptimal accuracy** on unseen data.

📊 **Contingency Matrix Analysis (10 Neurons):**
![image](https://github.com/user-attachments/assets/ce0248bc-16f4-42da-a519-d3f9af9d1dde)

These confusion matrices provide a **detailed view of the classification accuracy** during both training and testing phases of the neural network.

In this particular case:
* The **overall test accuracy is 83.7%**,  
* Which means the **error rate is 16.3%**, → This result is **unsatisfactory** and indicates that the network did not achieve the desired performance level.

>:arrow_left: [**Back to *CONTENT***](#content)

#### Second Variant: 200 Neurons
![image](https://github.com/user-attachments/assets/4dd22d08-9d29-4342-bd3c-fc8924cfa05a)

#### Key Elements of the Chart:
:large_blue_circle: *Blue Line (Training Data Error):*
   * The training error **drops sharply during the first few epochs**, indicating that the model can quickly learn.
   * As training progresses, the network reaches a very low error value of **8.8601e-06 at epoch 98**, marked on the graph.
   * The curve demonstrates that a network with **200 neurons has high learning capacity** and can capture complex patterns in the data. However, such a low error may also signal a **risk of overfitting**.

:red_circle: *Red Line (Testing Data Error):*
   * The test error shows gradual improvement, though **it doesn't decline as steeply as the training curve**.
   * The **stability of the test error** suggests the model still retains some generalization ability — although the network size might now be **excessive**, providing only marginal benefit.

📊 **Contingency Matrix (200 Neurons):**
![image](https://github.com/user-attachments/assets/73fe01d6-74c8-4e68-9a5a-2c7d9bb727e2)

These confusion matrices provide a **detailed view of classification accuracy** during both training and testing of the neural network.

In this particular case:
* The **overall test accuracy is 94.9%**,
* Resulting in an **error rate of 5.1%**.

Although this performance is **strong**, it falls just short of the required threshold, meaning the model is **good but not sufficient** for the task’s defined accuracy goals.

>:arrow_left: [**Back to *CONTENT***](#content)

### :pill: Testing with One Sample per Digit
![image](https://github.com/user-attachments/assets/a367cf83-071a-40ae-a789-dbb68db9835a)

> :exclamation: The result on the screenshot is **in Slovak**, while the output in the published code is **in English**.
> The left column shows the **actual digit**, and the right column shows the **predicted digit**.

>:arrow_left: [**Back to *CONTENT***](#content)
