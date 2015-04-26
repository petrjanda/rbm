package nn

import nn.conf.NNConf
import nn.ds.DataSet
import nn.fn.act.Sigmoid
import nn.fn.learn.ConstantRate
import nn.fn.loss.CrossEntropy
import nn.trainer.RBMTrainer
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.concurrent.ScalaFutures
import org.scalatest.{FlatSpec, Matchers}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.language.postfixOps
import scala.concurrent.duration._
import scala.util.{Failure, Success}

class RBMTest extends FlatSpec with Matchers with ScalaFutures {
  implicit val defaultPatience = PatienceConfig(timeout = 10 seconds)

  it should "get reasonable loss after 10 learning iterations" in {
    val input = Nd4j.ones(20, 2)
    val dataSet = DataSet(input, input)

    implicit val rng = new MersenneTwister(123)

    val conf = NNConf(
      activation = Sigmoid,
      loss = CrossEntropy
    )

    val s = System.currentTimeMillis()
    val rbm = RBMTrainer(
      epochs = 100,
      miniBatchSize = 10,
      parallel = 1,
      learningRate = ConstantRate(.5)
    ).train(RBM(2, 1, conf), dataSet)

//    rbm.onComplete {
//      case Success(k) =>
//      case Failure(ex) => ex.printStackTrace()
//    }

    whenReady(rbm) { rbm =>
      rbm.loss(input) should be < .01
    }
  }
}