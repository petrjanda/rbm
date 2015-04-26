package nn.trainer

import nn.ds.DataSet
import nn.fn.learn.LearningFunction
import org.apache.commons.math3.random.MersenneTwister
import nn.RBM

import scala.concurrent.{ExecutionContext, Future}

case class RBMTrainer(epochs: Int, miniBatchSize: Int, learningRate:LearningFunction) {
  def train(rbm: RBM, dataSet: DataSet)(implicit ec:ExecutionContext, rng:MersenneTwister): Future[RBM] = {
    val iterations = dataSet.numExamples * epochs / miniBatchSize

    dataSet.miniBatches(miniBatchSize).take(iterations).zipWithIndex.foldLeft(Future { rbm }) { (rbm, i) =>
      rbm.flatMap { nn =>
        ContrastiveDivergence.diff(nn, i._1, 1).map { gradient =>
          nn.update(gradient.rate(learningRate(i._2)))
        }
      }
    }
  }
}
