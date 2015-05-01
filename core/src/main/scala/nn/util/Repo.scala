package nn.util

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

class Repo(dir:String) {
  def save[T](obj: T, path: String) = {
    val os = new ObjectOutputStream(new FileOutputStream(dir + path))
    try {
      os.writeObject(obj)
    } finally {
      os.close()
    }
  }

  def load[T](path: String): T = {
    val is = new ObjectInputStream(new FileInputStream(dir + path))
    try {
      is.readObject().asInstanceOf[T]
    } finally {
      is.close()
    }
  }
}

object Repo {
  trait Writable {

  }
}
