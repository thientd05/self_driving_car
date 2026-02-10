## Explanation of Model Components

### 1. **BatchNormalization**
**BatchNormalization** is a technique that stabilizes neural-network training by normalizing the outputs of each layer so that each batch has zero mean and unit variance.

Normalization formula:
$$
\hat{x} = \frac{x - \mu}{\sigma} 
$$
- \( \mu \): Mean of the batch outputs.
- \( \sigma \): Standard deviation of the batch outputs.

After normalization, **BatchNormalization** applies learnable **scale** (\(\gamma\)) and **shift** (\(\beta\)) parameters:
$$
y = \gamma \hat{x} + \beta
$$
- **Scale (\(\gamma\))**: Controls the spread of the output distribution.
- **Shift (\(\beta\))**: Re-centers outputs to the desired range.

### 2. **Dropout**
**Dropout** is a regularization technique that mitigates overfitting by randomly zeroing a percentage of units during training, preventing the network from relying too heavily on any one neuron.

Dropout behavior:
- In each training step, with dropout rate \( p \), each output unit \( x \) is multiplied by a random mask:
$$
x_{\text{dropout}} = x \cdot \text{mask}
$$
- **mask** is a random vector containing 0s and 1s, where the probability of 1 is \( 1 - p \).

### 3. **He Normal Initialization (`he_normal`)**
**He Normal** initialization is tailored for networks using **ReLU** activations to reduce vanishing gradients.

Initialization rule:
$$
W \sim \mathcal{N}(0, \frac{2}{n_{\text{input}}})
$$
- Weights are sampled from a normal distribution with zero mean and variance \( \frac{2}{n_{\text{input}}} \), where \( n_{\text{input}} \) is the number of inputs to the layer.

### 4. **Glorot Uniform Initialization (`glorot_uniform`)**
**Glorot Uniform** (a.k.a. Xavier Uniform) works well for networks using **sigmoid** or **tanh** activations, keeping signal variance consistent across layers.

Initialization rule:
$$
W \sim \mathcal{U} \left( -\sqrt{\frac{6}{n_{\text{input}} + n_{\text{output}}}}, \sqrt{\frac{6}{n_{\text{input}} + n_{\text{output}}}} \right)
$$
- Weights are drawn from a uniform range \( \left[ -\text{limit}, \text{limit} \right] \), where:
  $$
  \text{limit} = \sqrt{\frac{6}{n_{\text{input}} + n_{\text{output}}}}
  $$
- \( n_{\text{input}} \) and \( n_{\text{output}} \) are the numbers of units in the input and output layers, respectively.
