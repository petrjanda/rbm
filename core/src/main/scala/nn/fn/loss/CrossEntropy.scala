package nn.fn.loss

import nn.fn.loss
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions

object CrossEntropy extends loss.LossFunction {
  def apply(labels: INDArray, output: INDArray, l2: Double, useRegularization: Boolean) = {
    LossFunctions.score(labels, LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY, output, l2, useRegularization)
  }
}
