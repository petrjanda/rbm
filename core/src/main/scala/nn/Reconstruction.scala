package nn

import nn.fn.loss.LossFunction
import org.nd4j.linalg.api.ndarray.INDArray

trait Reconstruction {
  def loss: LossFunction

  private[nn] def propUp(x: INDArray): INDArray

  private[nn] def propDown(x: INDArray): INDArray

  private[nn] def reconstruct(x: INDArray): INDArray = {
    propDown(propUp(x))
  }

  /**
   * Loss function, given the input matrix.
   *
   * @param x Input matrix
   * @return The loss coeficient
   */
  def loss(x: INDArray): Double =
    math.abs(loss(x, reconstruct(x)))
}
