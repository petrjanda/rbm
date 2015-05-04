package nn.trainer.fnn

import com.typesafe.scalalogging
import ds.MNIST
import nn.ds.DataSet
import nn.fn.act.Sigmoid
import nn.fn.learn.ConstantRate
import nn.fn.loss.CrossEntropy
import nn.util.{Matrix2Png, Repo}
import nn.{HiddenLayer, MultiLayerFNN}
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.concurrent.ScalaFutures
import org.scalatest.{FlatSpec, Matchers}
import org.slf4j.LoggerFactory

class FNNTrainerTest extends FlatSpec with Matchers with ScalaFutures {
  it should "do an iteration of back propagation" in {
    implicit val logger = scalalogging.Logger(LoggerFactory.getLogger("test"))
    implicit val rgn = new MersenneTwister(123)

    val layers = List(HiddenLayer(784, 10, activation = Sigmoid, loss = CrossEntropy))
//    val fnn = MultiLayerFNN(layers)

//    val ds = DataSet(
//      Nd4j.ones(1, 3),
//      Nd4j.zeros(1, 1)
//    )
    val mnist = MNIST.read("data/mnist/train-labels-idx1-ubyte", "data/mnist/train-images-idx3-ubyte", Some(1))
    val ds = DataSet(mnist._1.transpose, mnist._2.transpose)

    val trainer = FNNTrainer(
      epochs = 100,
      miniBatchSize = 1,
      learningRate = ConstantRate(.05)
    )

//    val repo = new Repo("src/test/resources/nets/fnn-mnist.o")

    val fnn = trainer.train(MultiLayerFNN(layers), ds)

    Range(0, 1).map { i =>
      Matrix2Png(ds.inputs.getRow(i).dup.reshape(28, 28)).write(s"in-$i.png")
    }

    println(fnn.loss(ds))

    import org.nd4j.linalg.ops.transforms.Transforms._

    println(fnn.propUp(ds.inputs).getRow(0).dup.data.asDouble.toList)
    println(ds.labels.getRow(0).dup.data.asDouble.toList)
  }

}
