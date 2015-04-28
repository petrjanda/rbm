package nn

import nn.fn.act.ActivationFunction
import nn.fn.loss.LossFunction
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.sampling.Sampling._

object HiddenLayer {
  def apply(numVisible: Int, numHidden: Int, activation: ActivationFunction, loss: LossFunction)(implicit rng:MersenneTwister) = {
    val W = normal(rng, Nd4j.zeros(numVisible, numHidden), .5)
    val h = Nd4j.zeros(numHidden)

    new HiddenLayer(W, h, activation, loss)
  }
}

class HiddenLayer(W: INDArray, h: INDArray, activation: ActivationFunction, loss: LossFunction) {
  lazy val numInputs = W.rows()

  lazy val numOutputs = W.columns()

  /**
   * Propagate the given input down.
   *
   * @param x
   * @return
   */
  private[nn] def propUp(x: INDArray): INDArray =
    activation(x.mmul(W).addRowVector(h))
}