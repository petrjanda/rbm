import nn.fn.act.ActivationFunction
import org.nd4j.linalg.api.ndarray.INDArray

case class LayerGradients(activation:ActivationFunction) {
  def errorGradients(target: INDArray, output: INDArray): INDArray = {
    val outErrorDerivative = activation.derivative(output)

    gradient(output, derivatives(target, outErrorDerivative))
  }

  private def gradient(output: INDArray, derivative: INDArray): INDArray =
    output.mmul(derivative.transpose)

  private def derivatives(priorDerivative: INDArray, output: INDArray) = {
    activation.derivative(output).mul(priorDerivative)
  }
}