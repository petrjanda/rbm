package nn

import nn.fn.act.ActivationFunction
import org.nd4j.linalg.api.ndarray.INDArray

trait NN[T] {
  def loss(x: INDArray): Double

  private[nn] def update(grad: T): NN[T]

  def activation: ActivationFunction
}
