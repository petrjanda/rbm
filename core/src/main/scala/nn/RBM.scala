package nn

import nn.fn.act.ActivationFunction
import nn.fn.loss.LossFunction
import nn.trainer.RBMGradient
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.sampling.Sampling._

object RBM {
  def apply(numVisible: Int, numHidden: Int, activation: ActivationFunction, loss: LossFunction)(implicit rng:MersenneTwister) = {
    val W = normal(rng, Nd4j.zeros(numVisible, numHidden), .5)
    val v = Nd4j.zeros(numVisible)
    val h = Nd4j.zeros(numHidden)

    new RBM(W, v, h, activation, loss)
  }

  def apply(W: INDArray, v: INDArray, h: INDArray, activation: ActivationFunction, loss: LossFunction) =
    new RBM(W, v, h, activation, loss)
}

class RBM(W: INDArray, v: INDArray, h: INDArray, activation: ActivationFunction, loss: LossFunction) extends HiddenLayer(W, h, activation, loss) {
  /**
   * Propagate the given hidden layer value up.
   *
   * @param x
   * @return
   */
  private[nn] def propDown(x: INDArray): INDArray =
    activation(x.mmul(W.transpose).addRowVector(v))

  /**
   * Loss function, given the input matrix.
   *
   * @param x Input matrix
   * @return The loss coeficient
   */
  def loss(x: INDArray): Double =
    math.abs(loss(x, reconstruct(x)))

  /**
   * Reconstruct the given input.
   *
   * @param x Input matrix to be reconstructed
   * @return Output (reconstruction) matrix
   */
  def reconstruct(x: INDArray): INDArray =
    propDown(propUp(x))

  private[nn] def update(grad:RBMGradient) =
    new RBM(W.add(grad.W), v.add(grad.v), h.add(grad.h), activation, loss)

  override def toString =
    s"RBM(W:$W, v:$v, h:$h)"
}