package nn.trainer.fnn

import com.typesafe.scalalogging.Logger
import nn.MultiLayerFNN
import nn.ds.DataSet
import nn.fn.learn.LearningFunction
import nn.trainer.Gradients
import org.apache.commons.math3.random.MersenneTwister

case class FNNTrainer(epochs: Int, miniBatchSize: Int, learningRate:LearningFunction) {
  def train(fnn: MultiLayerFNN, dataSet: DataSet)(implicit log:Logger, rng:MersenneTwister): MultiLayerFNN = {
    val iterations = dataSet.numExamples * epochs / miniBatchSize

    dataSet.miniBatches(miniBatchSize).take(iterations).zipWithIndex.foldLeft(fnn) { (fnn, i) =>
      if(i._2 % 100 == 0) log.info(s"--> ${i._2}, ${fnn.loss(dataSet)}")
      
      fnn.update(
        Gradients(fnn).errorGradients(i._1).map(_.rate(learningRate(i._2)))
      )
    }
  }
}
