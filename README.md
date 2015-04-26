## Restricted Bolzmann Machine (in Scala)

Motivation of this repo is to provide simple, easy to follow implementation of Restricted Bolzmann Machine in idiomatic
scala code.

Initially I got inspired by nice example of Feed Forward neural net demonstrated on MNISt dataset (https://github.com/mlehman/mnist-nn-example),
which nicely shows capability of NN in supervised environment. When I went ahead to explore unsupervised learning realm,
I didn't find any implementation except example from https://github.com/yusugomori/DeepLearning/blob/master/scala/RBM.scala.
Even though author managed to port RBM algorithm across different languages the Scala implementation isn't very idiomatic
and a quite a lot of code is unnecessary complicated - which hides the simplicity of used algorithms. Never the less it
gave me excellent insight into RBMs in the first stage of my exploration.