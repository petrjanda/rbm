package nn.trainer

import com.typesafe.scalalogging.Logger
import nn.NN
import nn.ds.DataSet
import nn.fn.learn.LearningFunction
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray

abstract class Trainer[T <: NN[S], S <: RBMGradient](epochs: Int, miniBatchSize: Int, learningRate:LearningFunction) {
  def train(rbm: T, dataSet: DataSet)(implicit log:Logger, rng:MersenneTwister): T = {
    val iterations = dataSet.numExamples * epochs / miniBatchSize

    dataSet.miniBatches(miniBatchSize).take(iterations).zipWithIndex.foldLeft(rbm) { (rbm, i) =>
      log.info(s" ${i._2}, ${rbm.loss(dataSet.inputs)}")

      rbm.update(train(rbm, i._1).rate[S](learningRate(i._2))).asInstanceOf[T]
    }
  }

  def train(rbm: T, inputs: INDArray)(implicit log:Logger, rng:MersenneTwister): S
}
