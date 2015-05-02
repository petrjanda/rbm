package nn.fn.act

import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray

trait ActivationFunction extends Serializable {
  def apply(x: INDArray)(implicit rng:MersenneTwister): INDArray

  def derivative(x: INDArray): INDArray
}
