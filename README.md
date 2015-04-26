## Restricted Bolzmann Machine (in Scala)

Motivation of this repo is to provide simple, easy to follow implementation of Restricted Bolzmann Machine in idiomatic
scala code.

### Inspired by

Initially I got inspired by nice example of Feed Forward neural net demonstrated on MNISt dataset (https://github.com/mlehman/mnist-nn-example),
which nicely shows capability of NN in supervised environment. Second source of inspiration was RBM implementation in Scala - https://github.com/yusugomori/DeepLearning/blob/master/scala/RBM.scala (probably port of Python code).
Even though its not written in idiomatic scala, it gave me initial insight into how different algorithms fit together. That said,
I saw it was a quick port of code from Python therefore there was plenty of opportunities to clean it up and make it more simple.

After initial playing with both repos I've decided to start with the clean slate and write RBM implementation myself.

### Architecture

Matrices - To achieve reasonable performance, all matrix operations are based on Nd4j library (http://nd4j.org/). Nd4j is Linear Algebra
implementation in Java, which is powering comprehensive NN package - Deeplearning4j. As its an abstraction over several
different implementation it offers an opportunity to swap various "backends" including JCublas backend offering native GPU support (http://nd4j.org/gpu_native_backends.html).

Futures - As I would like to experiment with several techniques for scaling out, all the computationally intense operations
are wrapped in Futures with hope I'll be able to try different techniques to parallelize the computations (Futures, Akka, Akka Streams, Spark?).

Immutability - Where possible I do try to keep principles of FP and keep all data immutable. As an example you can see that
trainer never modifies a net, but instead creates an instance of a new net as a result of training iteration. In general
I would like to keep this pattern.




