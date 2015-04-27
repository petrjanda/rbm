package nn.fn.loss

import nn.fn.loss
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms._

object CrossEntropy extends loss.LossFunction {
  def apply(y: INDArray, a: INDArray) = {
    - y.mul(log(a)).add(y.neg.add(1).mul(log(a.neg.add(1))))
      .sum(1).mean(Integer.MAX_VALUE).getDouble(0)
  }
}
