package nn.trainer

import com.typesafe.scalalogging.Logger
import nn.RBM
import nn.ds.DataSet
import nn.fn.learn.LearningFunction
import org.apache.commons.math3.random.MersenneTwister

import scala.concurrent
import scala.concurrent.{ExecutionContext, Future}

case class RBMTrainer(epochs: Int, miniBatchSize: Int, learningRate:LearningFunction) {
  def train(rbm: RBM, dataSet: DataSet)(implicit log:Logger, rng:MersenneTwister, ec:ExecutionContext): Future[RBM] = {
    val iterations = dataSet.numExamples * epochs / miniBatchSize

    dataSet.miniBatches(miniBatchSize).take(iterations).zipWithIndex.foldLeft(Future { rbm }) { (rbm, i) =>
//      if(i._2 % 100 == 0) log.info(s"--> ${i._2}, ${rbm.loss(dataSet.inputs)}")
      rbm.flatMap { rbm =>
        val diff = ContrastiveDivergence.diff(rbm, i._1, 1)

        diff.map(d => rbm.update(d.rate(learningRate(i._2))))
      }
    }
  }
}
