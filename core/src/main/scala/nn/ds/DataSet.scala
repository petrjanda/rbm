package nn.ds

import org.nd4j.linalg.api.ndarray.INDArray

case class DataSet(inputs:INDArray, labels:INDArray) {
  lazy val numExamples = inputs.rows()

  def miniBatches(batchSize: Int) =
    Stream.continually(0 until numExamples).flatten.grouped(batchSize).map {
      rows => inputs.getRows(rows.toArray)
    }
}
