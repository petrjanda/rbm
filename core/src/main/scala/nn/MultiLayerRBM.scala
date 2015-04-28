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


class MultiLayerRBM(override val layers:List[RBM], val loss: LossFunction) extends MultiLayerNN(layers, loss) with Reconstruction {
  private[nn] def propDown(x: INDArray): INDArray = {
    layers.foldRight(x) {
      case (layer, x) => layer.propDown(x)
    }
  }
}