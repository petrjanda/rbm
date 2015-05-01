package ds

import java.io.{DataInputStream, FileInputStream}

import org.jblas.DoubleMatrix
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ArrayBuffer

/**
 * This class implements a reader for the MNIST dataset of handwritten digits. The dataset is found
 * at http://yann.lecun.com/exdb/mnist/.
 *
 * @author Gabe Johnson <johnsogg@cmu.edu>
 */
object MNIST {
  def read(labelfile: String, imagefile : String, max: Option[Int] = None): (INDArray, INDArray) = {
    val labels = new DataInputStream(new FileInputStream(labelfile))
    val images = new DataInputStream(new FileInputStream(imagefile))
    var magicNumber = labels.readInt()

    if (magicNumber != 2049) {
      System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)")
      System.exit(0)
    }

    magicNumber = images.readInt()

    if (magicNumber != 2051) {
      System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)")
      System.exit(0)
    }

    var numLabels = labels.readInt()
    var numImages = images.readInt()
    val numRows = images.readInt()
    val numCols = images.readInt()

    if(max.isDefined) {
      numLabels = max.get
      numImages = max.get
    }

    val image = Nd4j.create(numCols * numRows, numImages)
    val label = Nd4j.create(1, numLabels)

    if (numLabels != numImages) {
      System.err.println("Image file and label file do not contain the same number of entries.")
      System.err.println("  Label file contains: " + numLabels)
      System.err.println("  Image file contains: " + numImages)
      System.exit(0)
    }

    var numLabelsRead = 0
    var numImagesRead = 0

    while (labels.available() > 0 && numLabelsRead < numLabels) {
      label.put(numImagesRead, 1, labels.readByte().toDouble)

      numLabelsRead += 1

      val temp = ArrayBuffer[Double]()
      for (colIdx <- 0 until numCols) {

        for (rowIdx <- 0 until numRows) {
          temp += images.readUnsignedByte().toDouble / 255
        }
      }

      image.putColumn(numImagesRead, Nd4j.create(temp.toArray, Array(1, 784)))
      temp.clear()

      numImagesRead += 1

      if(numImagesRead % 100 == 0) {
        print(".")
      }
    }

    println()
    (image, label)
  }
}