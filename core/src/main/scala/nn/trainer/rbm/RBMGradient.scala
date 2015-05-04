package nn.trainer.rbm

import org.nd4j.linalg.api.ndarray.INDArray

case class RBMGradient(W: INDArray, v: INDArray, h: INDArray) {
  def rate(r: Double) = RBMGradient(W.mul(r), v.mul(r), h.mul(r))
}
