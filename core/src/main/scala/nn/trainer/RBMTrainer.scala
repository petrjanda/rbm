package nn.trainer

import nn.ds.DataSet
import nn.fn.learn.LearningFunction
import org.apache.commons.math3.random.MersenneTwister
import nn.RBM

import scala.concurrent.{ExecutionContext, Future}

case class RBMTrainer(epochs: Int, miniBatchSize: Int, parallel: Int, learningRate:LearningFunction) {
  def train(rbm: RBM, dataSet: DataSet)(implicit ec:ExecutionContext, rng:MersenneTwister): Future[RBM] = {
    val iterations = dataSet.numExamples * epochs / (miniBatchSize * parallel)

    dataSet.miniBatches(miniBatchSize).grouped(parallel).take(iterations).zipWithIndex.foldLeft(Future { rbm }) { (rbm, i) =>
      val batches = i._1
      val iteration = i._2
      rbm.flatMap { nn =>
        Future.sequence(batches.map(ContrastiveDivergence.diff(nn, _, 1))).map { gradients =>
          val gradient = gradients.reduceLeft { _ avg _ }

          if(iteration % (iterations / 100) == 0) println(s" iteration: $iteration, loss: ${nn.loss(dataSet.inputs)}")

          nn.update(gradient.rate(learningRate(iteration)))
        }
      }
    }
  }
}
