package nn.fn.act

import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object Linear extends ActivationFunction {
  def apply(x: INDArray)(implicit rng: MersenneTwister): INDArray = x

  def derivative(x: INDArray) = Nd4j.zeros(x.rows, x.columns)
}
