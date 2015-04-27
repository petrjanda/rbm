package nn.trainer

import com.typesafe.scalalogging.Logger
import nn.RBM
import nn.fn.learn.LearningFunction
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray

case class RBMTrainer(epochs: Int, miniBatchSize: Int, learningRate:LearningFunction) extends Trainer[RBM, RBMGradient](epochs, miniBatchSize, learningRate) {
  def train(rbm: RBM, inputs: INDArray)(implicit log:Logger, rng:MersenneTwister) =
    ContrastiveDivergence.diff(rbm, inputs, 1)
}
