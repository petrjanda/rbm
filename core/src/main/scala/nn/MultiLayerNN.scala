package nn

import nn.fn.loss.LossFunction
import org.nd4j.linalg.api.ndarray.INDArray

class MultiLayerNN(val layers:List[HiddenLayer]) {
  lazy val numInputs = layers.head.numInputs

  lazy val numOutputs = layers.last.numOutputs

  private[nn] def propUp(x: INDArray): INDArray = {
    layers.foldLeft(x) {
      case (x, layer) => layer.propUp(x)
    }
  }
}
