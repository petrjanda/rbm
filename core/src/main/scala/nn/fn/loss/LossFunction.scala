package nn.fn.loss

import org.nd4j.linalg.api.ndarray.INDArray

trait LossFunction extends Serializable {
  def apply(labels: INDArray, output: INDArray): Double
}
