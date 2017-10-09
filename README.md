# clj-autograd

These are some experiments to bring
a
[pytorch](http://pytorch.org/docs/master/index.html)-like
[autograd](http://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/) to
Clojure. This is not primarily a neural network library, but rather a means of
abstraction to calculate gradients for scientific computing. `clj-autograd` uses
neanderthal at the moment, because we want to be as fast as possible with
matrices to allow to build competitive deep learning and Bayesian inference on
top.


## Usage

Look at the tests for now.

## TODO

- polymorphic product for scalar, vector, matrix, tensor
- identity of Variable? yes, as atom for inplace ops
- trace shapes
- 1. demo
  linear regression DONE
  gradient check DONE
- basic operations to classify MNIST:
- loss: sub, pow DONE
- forward with (mini)batches:
  + mmul
- activations:
- sigmoid DONE
- requires_grad DONE
- optimizer DONE
- lazyness?
- figure out how to do in-place ops?



## License

Copyright © 2017 Christian Weilbach

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
