package nn

import nn.fn.act.ActivationFunction
import nn.fn.loss.LossFunction
import nn.trainer.RBMGradient
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.sampling.Sampling._

object RBM {
  def apply(numVisible: Int, numHidden: Int, visibleActivation: ActivationFunction, hiddenActivation: ActivationFunction, loss: LossFunction)(implicit rng:MersenneTwister) = {
    val W = normal(rng, Nd4j.zeros(numVisible, numHidden), .5)
    val v = Nd4j.zeros(numVisible)
    val h = Nd4j.zeros(numHidden)

    println(W)

    new RBM(W, v, h, visibleActivation, hiddenActivation, loss)
  }

  def apply(W: INDArray, v: INDArray, h: INDArray, visibleActivation: ActivationFunction, hiddenActivation: ActivationFunction, loss: LossFunction)(implicit rng:MersenneTwister) =
    new RBM(W, v, h, visibleActivation, hiddenActivation, loss)
}

class RBM(W: INDArray, val v: INDArray, val h: INDArray, val visibleActivation: ActivationFunction, val hiddenActivation: ActivationFunction, val loss: LossFunction)(implicit rng:MersenneTwister) extends HiddenLayer(W, h, hiddenActivation, loss) with Reconstruction {
  /**
   * Propagate the given hidden layer value up.
   *
   * @param x
   * @return
   */
  private[nn] def propDown(x: INDArray): INDArray =
    visibleActivation(x.mmul(W.transpose).addRowVector(v))

  private[nn] def update(grad:RBMGradient) =
    new RBM(W.add(grad.W), v.add(grad.v), h.add(grad.h), visibleActivation, hiddenActivation, loss)

  override def toString =
    s"RBM(W:$W, v:$v, h:$h)"
}