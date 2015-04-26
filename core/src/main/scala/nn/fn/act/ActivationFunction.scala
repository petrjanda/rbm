package nn.fn.act

import org.nd4j.linalg.api.ndarray.INDArray

trait ActivationFunction {
  def apply(x: INDArray): INDArray
}
