package nn.util

import com.sksamuel.scrimage.{Image, RGBColor}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms._

object Matrix2Png {
  def apply(m: INDArray) = {
    import org.nd4j.linalg.ops.transforms.Transforms._

    val max = abs(m).data.asDouble.max

    val data = m.div(max).data.asDouble.map { i => {
      val c = 127 + (127 * i).toInt

      RGBColor(c, c, c, 255).toInt
    } }

    Image(m.columns, m.rows, data.toArray)
  }
}