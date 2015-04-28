package nn

import nn.fn.act.ActivationFunction
import nn.fn.loss.LossFunction
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray

object MultiLayerRBM {
  def apply(numVisible: Int, numHidden: List[(Int, ActivationFunction)], loss: LossFunction)(implicit rng:MersenneTwister) = {
    val layers = numHidden.foldLeft(List(RBM(numVisible, numHidden.head._1, numHidden.head._2, loss))) { (previous, i) =>
      previous :+ RBM(previous.last.numOutputs, i._1, i._2, loss)
    }

    new MultiLayerRBM(layers, loss)
  }

  def apply(layers:List[RBM], loss: LossFunction)(implicit rng:MersenneTwister) = {
    new MultiLayerRBM(layers, loss)
  }
}

class MultiLayerRBM(val layers:List[RBM], loss: LossFunction) {
  lazy val numInputs = layers.head.numInputs

  lazy val numOutputs = layers.last.numOutputs

  private[nn] def propUp(x: INDArray): INDArray = {
    layers.foldLeft(x) {
      case (x, layer) => layer.propUp(x)
    }
  }

  private[nn] def propDown(x: INDArray): INDArray = {
    layers.foldRight(x) {
      case (layer, x) => layer.propDown(x)
    }
  }

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