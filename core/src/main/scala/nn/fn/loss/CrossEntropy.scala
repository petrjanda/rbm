package nn.fn.loss

import nn.fn.loss
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms._

object CrossEntropy extends loss.LossFunction {
  def apply(y: INDArray, a: INDArray) = {
    // −Y logfθ(X)−(1−Y)log(1−fθ(X))

    -y.mul(log(a)).add(y.neg.add(1).mul(log(a.neg.add(1))))
      .sum(1).mean(Integer.MAX_VALUE).getDouble(0)
  }

  def derivative(x: INDArray, y: INDArray): INDArray =
    x.sub(y).muli(1.0 / y.columns)
}
