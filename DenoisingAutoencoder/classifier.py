import numpy as np
import matplotlib.pyplot as plt


class Dataset(object):
  """Dataset used in the example."""
  
  def __init__(self):
    self._training_data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    self._test_data = np.array([[1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]])

  @property
  def training(self):
    return self._training_data

  @property
  def testing(self):
    return self._test_data


def display(original, processed):
  """Displays two images one next to another."""
  fig = plt.figure()
  a = fig.add_subplot(1,2,1)
  a.set_title('Original')
  plt.imshow(original)

  a = fig.add_subplot(1,2,2)
  a.set_title('Processed')
  plt.imshow(processed)

  plt.show()


def sigmoid(x):
  """Sigmoid function."""
  return 1. / (1 + np.exp(-x))


class DenoisingAutoencoder(object):
  """Autoencoder capable of removing noise from the input."""

  def __init__(self, n_hidden=3, n_output=2):
    """Constructor."""
    a = 1. / n_output
    initial_W = np.array(np.random.uniform(
        low=-a,
        high=a,
        size=(n_output, n_hidden)))

    self._h_bias = np.zeros(n_hidden)
    self._v_bias = np.zeros(n_output)
    self._W = initial_W
    self._W_prime = self._W.T

  def _get_noisy_input(self, input, noise_level):
    return np.random.binomial(size=input.shape, n=1, p=1-noise_level) * input

  def _get_hidden(self, input):
    return sigmoid(np.dot(input, self._W) + self._h_bias)

  def _get_reconstructed_input(self, input):
    return sigmoid(np.dot(input, self._W_prime) + self._v_bias)

  def train(self, x, learning_rate=0.1, noise_level=0.1):
    """Trains the autoencoder.
    
    Args:
      x: Input data for training. A 2D array of shape (n x m).
    """
    tilde_x = self._get_noisy_input(x, noise_level)
    y = self._get_hidden(tilde_x)
    z = self._get_reconstructed_input(y)

    magic_sauce = y * (1-y)

    L_vbias = x - z
    L_hbias = np.dot(L_vbias, self._W) * magic_sauce
    L_W = np.dot(tilde_x.T, L_hbias) + np.dot(L_vbias.T, y)

    self._W += learning_rate * L_W
    self._h_bias += learning_rate * np.mean(L_hbias, axis=0)
    self._v_bias += learning_rate * np.mean(L_vbias, axis=0)

  def reconstruct(self, x):
    """Reconstructs the input from the given output."""
    y = self._get_hidden(x)
    z = self._get_reconstructed_input(y)
    return z

def main():
  training_epochs = 20

  data = Dataset()

  model = DenoisingAutoencoder(n_hidden=5, n_output=data.training.shape[1])
  for epoch in xrange(training_epochs):
    model.train(data.training)

  denoised_data = model.reconstruct(data.testing)
  display(data.testing, denoised_data)

if __name__ == '__main__':
  main()
