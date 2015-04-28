package nn.fn.loss

import org.nd4j.linalg.api.ndarray.INDArray

trait LossFunction {
  def apply(labels: INDArray, output: INDArray): Double

  def derivative(y: INDArray, a: INDArray): INDArray
}
