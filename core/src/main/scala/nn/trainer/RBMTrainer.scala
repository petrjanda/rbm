package nn.trainer

import com.typesafe.scalalogging.Logger
import nn.RBM
import nn.ds.DataSet
import nn.fn.learn.LearningFunction
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray

import scala.concurrent
import scala.concurrent.{ExecutionContext, Future}

case class RBMTrainer(epochs: Int, miniBatchSize: Int, learningRate:LearningFunction)(implicit log:Logger, rng:MersenneTwister, ec:ExecutionContext) {
  def train(rbm: RBM, dataSet: DataSet): Future[RBM] = {
    val iterations = dataSet.numExamples * epochs / miniBatchSize

    dataSet.miniBatches(miniBatchSize).take(iterations).grouped(1).zipWithIndex.foldLeft(Future { rbm }) { (rbm, i) =>
      rbm.flatMap { rbm =>
        trainBatches(rbm, i._1).map(d => {
          if(i._2 % 100 == 0) log.info(s"--> ${i._2}, ${rbm.loss(dataSet.inputs)}")

          rbm.update(d.rate(learningRate(i._2)))
        })
      }
    }
  }

  /**
   * Perform contrastive divergence algorithm on several mini batches in parallel. The calculated
   * gradients are averaged.
   *
   * @param rbm RBM instance.
   * @param batches Sequence of batches of training data
   * @return  RBM gradient as an average of gradients from individual batches
   */
  private def trainBatches(rbm: RBM, batches: Seq[INDArray]): Future[RBMGradient] = {
    val gradients = Future.sequence(batches.map(batch => ContrastiveDivergence.diff(rbm, batch, 1)))

    gradients.map { gradients =>
      gradients.reduce { (t, i) => RBMGradient(t.h.add(i.h), t.v.add(i.v), t.W.add(i.W)) }
    }
  }
}
