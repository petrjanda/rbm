package nn.ds

import org.nd4j.linalg.api.ndarray.INDArray

object DataSet {
  def apply(inputs: INDArray): DataSet =
    new DataSet(inputs, inputs)
}

case class DataSet(inputs:INDArray, val labels:INDArray) {
  lazy val numExamples = inputs.rows()

  def miniBatches(batchSize: Int) =
    Stream.continually(0 until numExamples).flatten.grouped(batchSize).map {
      rows => inputs.getRows(rows.toArray)
    }
}
