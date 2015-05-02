package nn

import nn.trainer.fnn.FNNGradient
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray

class MultiLayerFNN(val layers:List[HiddenLayer]) {
  lazy val numInputs = layers.head.numInputs

  lazy val numOutputs = layers.last.numOutputs

  private[nn] def propUp(x: INDArray): INDArray = {
    layers.foldLeft(x) {
      case (x, layer) => layer.propUp(x)
    }
  }

  private[nn] def propUp2(x: INDArray): List[INDArray] = {
    layers.scanLeft(x) {
      case (x, layer) => layer.propUp(x)
    }.tail
  }

  private[nn] def update(grad: Seq[FNNGradient])(implicit rng: MersenneTwister) =
    new MultiLayerFNN(
      layers.zip(grad).map { case (layer, grad) =>
        new HiddenLayer(layer.W.add(grad.W), layer.h, layer.activation, layer.loss)
      }
    )

  def loss(x: INDArray, y: INDArray): Double =
    loss(x, y)
}
