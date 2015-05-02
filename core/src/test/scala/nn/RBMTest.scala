package nn

import _root_.ds.MNIST
import com.typesafe._
import nn.ds.DataSet
import nn.fn.act.Sigmoid
import nn.fn.learn.ConstantRate
import nn.fn.loss.CrossEntropy
import nn.trainer.RBMTrainer
import org.apache.commons.math3.random.MersenneTwister
import org.scalatest.concurrent.ScalaFutures
import org.scalatest.time.{Seconds, Span}
import org.scalatest.{FlatSpec, Matchers}
import org.slf4j.LoggerFactory

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

class RBMTest extends FlatSpec with Matchers with ScalaFutures {
  implicit val logger = scalalogging.Logger(LoggerFactory.getLogger("test"))
  
  implicit val rng = new MersenneTwister(123)

  val trainer = RBMTrainer(
    epochs = 10,
    miniBatchSize = 25,
    learningRate = ConstantRate(0.05)
  )

  override implicit def patienceConfig = PatienceConfig(timeout = Span(60, Seconds))

  it should "get reasonable loss after learning with ones as targets" in {
    val mnist = MNIST.read("data/mnist/train-labels-idx1-ubyte", "data/mnist/train-images-idx3-ubyte", Some(1000))
    val ds = DataSet(mnist._1.transpose)

    val l1 = trainer.train(RBM(784, 500, Sigmoid, Sigmoid, CrossEntropy), ds)

    val rbm = Future.sequence(List(l1)).map(MultiLayerRBM(_, CrossEntropy))

    whenReady(rbm) { rbm =>
      println("deep: " + rbm.loss(ds.inputs))
    }
  }
}