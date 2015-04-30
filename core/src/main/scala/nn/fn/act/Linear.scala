package nn.fn.act

import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray

object Linear extends ActivationFunction {
  def apply(x: INDArray)(implicit rng: MersenneTwister): INDArray = x
}
