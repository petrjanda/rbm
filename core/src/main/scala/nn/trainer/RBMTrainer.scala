package nn.trainer

import com.typesafe.scalalogging.Logger
import nn.RBM
import nn.ds.DataSet
import nn.fn.learn.LearningFunction
import org.apache.commons.math3.random.MersenneTwister

case class RBMTrainer(epochs: Int, miniBatchSize: Int, learningRate:LearningFunction) {
  def train(rbm: RBM, dataSet: DataSet)(implicit log:Logger, rng:MersenneTwister): RBM = {
    val iterations = dataSet.numExamples * epochs / miniBatchSize

    dataSet.miniBatches(miniBatchSize).take(iterations).zipWithIndex.foldLeft(rbm) { (rbm, i) =>
      if(i._2 % 100 == 0) log.info(s"--> ${i._2}, ${rbm.loss(dataSet.inputs)}")

      rbm.update(ContrastiveDivergence.diff(rbm, i._1, 1).rate(learningRate(i._2)))
    }
  }
}
