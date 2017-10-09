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

To build a logistic regression model and do gradient descent with the automatic
gradient, you can do something like this:

~~~clojure
(let [X (tape (trans (dge 2 3 [5 2 -1 0 5 2])))
        Y (tape (dv [1 0 1]))
        c (tape 1 true)
        b (tape (dv [1 1]) true)#_(show-grad )]
    (loop [i 1000]
      (when (pos? i)
        (let [Y* (sigmoid (add (mul X b) (broadcast-like c Y)))
              out (bcrossent Y* Y)
              grads ((:backward out) out 1)]
          (when (zero? (mod i 100))
            (prn "Loss:" @(:data out) ", b:"  @(:data b) ", c:" @(:data c)))
          (gd! grads)
          (recur (dec i)))))
    b)
~~~

Look at the tests for now.

## TODO

- polymorphic product for scalar, vector, matrix, tensor
- identity of Variable? yes, as atom for inplace ops
- trace shapes
- 1. demo
  + linear regression DONE
  + gradient check DONE
- basic operations to classify MNIST:
- loss: sub, pow DONE
- forward with (mini)batches:
  + mmul DONE
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
