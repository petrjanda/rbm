package nn.trainer

import nn._
import nn.ds.DataSet
import nn.fn.loss.LossFunction
import nn.trainer.fnn.FNNGradient
import org.nd4j.linalg.api.ndarray.INDArray

case class Gradients(nn:MultiLayerFNN) {
  val layers = nn.layers
  val objective:LossFunction = nn.layers.last.loss

  def errorGradients(data: DataSet): Seq[FNNGradient] = {
    val inputs = data.inputs
    val targets = data.labels
    val outputs = nn.propUp2(inputs)

    val outErrorDerivative = objective.derivative(outputs.last, targets)

    derivatives(outErrorDerivative, outputs).zipWithIndex.map {
      case (derivative, i) =>
        val x = if (i > 0) outputs(i - 1) else inputs

        FNNGradient(x.mmul(derivative.transpose))
    }
  }

  private def derivatives(outputErrorDerivative: INDArray, outputs: List[INDArray]) =
    (0 until layers.size).scanRight(outputErrorDerivative) {
      case (i, priorDerivative) =>
        val priorDerivativeWeighted = if (i < layers.size - 1) {
          layers(i + 1).W.mmul(priorDerivative)
        } else priorDerivative

        layers(i).activation.derivative(outputs(i)).mul(priorDerivativeWeighted)
    }
}

//package nn.trainers.backprop
//
//import nn.ds.DataSet
//import nn.fn.{ObjectiveFunction, WeightDecay}
//import nn.{FeedForwardNN, Layer, LayerState, FeedForwardNN$}
//import org.jblas.DoubleMatrix
//
//case class Gradients(nn:FeedForwardNN) {
//  val layers:List[Layer] = nn.layers
//  val objective:ObjectiveFunction = nn.objective
//  val weightDecay: WeightDecay = nn.weightDecay
//
//  def errorGradients(data: DataSet): Seq[DoubleMatrix] = {
//    val inputs = data.features
//    val targets = data.targets
//    val outputs = nn.propagate(inputs)
//
//    val outErrorDerivative = objective.derivative(outputs.last.activationOutput, targets)
//
//    derivatives(outErrorDerivative, outputs).zipWithIndex.map {
//      case (derivative, i) =>
//        val x = if (i > 0) outputs(i - 1).activationOutput else inputs
//
//        decayed(i, x.mmul(derivative.transpose))
//    }
//  }
//
//  private def decayed(i:Int, gradient: DoubleMatrix) =
//    if (i < layers.size) weightDecay.derivative(layers(i).weights, gradient) else gradient
//
//  private def derivatives(outputErrorDerivative:DoubleMatrix, outputs:List[LayerState]) =
//    (0 until layers.size).scanRight(outputErrorDerivative) {
//      case (i, priorDerivative) =>
//        val output = outputs(i)
//        val layer = layers(i)
//
//        val priorDerivativeWeighted = if (i < layers.size - 1) {
//          layers(i + 1).weights.mmul(priorDerivative)
//        } else priorDerivative
//
//        layer.activation.derivative(
//          output.compositionOutput.get,
//          output.activationOutput
//        ).mul(priorDerivativeWeighted)
//    }
//}