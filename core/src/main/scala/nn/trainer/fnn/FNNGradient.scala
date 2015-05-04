package nn.trainer.fnn

import org.nd4j.linalg.api.ndarray.INDArray

case class FNNGradient(W: INDArray) {
  def rate(r: Double) = FNNGradient(W.mul(r))
}
