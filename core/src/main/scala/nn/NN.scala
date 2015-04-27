package nn

import nn.trainer.RBMGradient
import org.nd4j.linalg.api.ndarray.INDArray

trait NN[T] {
  def loss(x: INDArray): Double

  private[nn] def update(grad: T): NN[T]
}
