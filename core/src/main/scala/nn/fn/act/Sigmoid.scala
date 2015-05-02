package nn.fn.act

import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms._

object Sigmoid extends ActivationFunction {
  def apply(x: INDArray)(implicit rng:MersenneTwister): INDArray = pow(exp(x.neg).add(1), -1)

  def derivative(y: INDArray): INDArray =
    y.mul(y.neg.addi(1))
}