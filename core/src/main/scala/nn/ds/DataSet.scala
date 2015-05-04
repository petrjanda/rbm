package nn.ds

import org.nd4j.linalg.api.ndarray.INDArray

object DataSet {
  def apply(inputs: INDArray): DataSet =
    new DataSet(inputs, inputs)
}

case class DataSet(inputs:INDArray, val labels:INDArray) {
  lazy val numExamples = inputs.rows

  def miniBatches(batchSize: Int): Iterator[DataSet] = {
    require(
      batchSize <= numExamples,
      s"Minibatch size $batchSize needs to be smaller or equal to data set size $numExamples!"
    )

    Stream.continually(0 until numExamples).flatten.grouped(batchSize).map {
      rows => DataSet(inputs.getRows(rows.toArray), labels.getRows(rows.toArray))
    }
  }
}
