package nn.trainer

import nn.ds.DataSet
import nn.fn.learn.LearningFunction
import org.apache.commons.math3.random.MersenneTwister
import nn.RBM

case class RBMTrainer(iterations: Int, learningRate:LearningFunction) {
  def train(rbm: RBM, dataSet: DataSet)(implicit rng:MersenneTwister): RBM =
    dataSet.miniBatches(1).take(iterations).zipWithIndex.foldLeft(rbm) { (rbm, i) =>
      rbm.update(ContrastiveDivergence.diff(rbm, i._1, 1).rate(learningRate(i._2)))
    }
}
