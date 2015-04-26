package nn.trainer

import org.nd4j.linalg.api.ndarray.INDArray

case class RBMGradient(W: INDArray, v: INDArray, h: INDArray) {
  def rate(r: Double) = RBMGradient(W.mul(r), v.mul(r), h.mul(r))

  def avg(that: RBMGradient) = RBMGradient(
    W.add(that.W).div(2),
    v.add(that.v).div(2),
    h.add(that.h).div(2)
  )
}
