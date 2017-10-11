(ns clj-autograd.core
  (:require [clojure.core.matrix :as m]
            [denisovan.core])
  (:use [uncomplicate.neanderthal core native]
        [uncomplicate.fluokitten core jvm]))

(m/set-current-implementation :neanderthal)

(comment
  (defn p+ ^double [^double x ^double y]
    (+ x y))

  (defn p* ^double [^double x ^double y]
    (* x y)))

(defn uuid []
  (java.util.UUID/randomUUID))

(defn tape
  ([data]
   (tape data false))
  ([data requires-grad?]
   {:data (atom data)
    :requires-grad? requires-grad?
    :id (uuid)}))

(defn zeros-like [x]
  (if (:data x)
    (tape (zeros-like @(:data x)))
    (m/zero-array (m/shape x))))

(defn constant-like [x c]
  (if (:data x)
    (tape (constant-like @(:data x) c))
    (m/add (m/zero-array (m/shape x)) c)))

(defn ones-like [x]
  (constant-like x 1.0))

(defn- back-fn [a]
  (or (:backward a) #(assoc %1 :grad %2)))

(defn broadcast-like [a b]
  (let [data (atom (m/scale (ones-like @(:data b)) @(:data a) ))]
    {:data data
     :id (uuid)
     :children [a]
     :requires-grad? (:requires-grad? a)
     :backward (fn [{:keys [children] :as node} grad]
                 (-> node
                     (assoc :grad grad)
                     (update :children
                             (fn [[a]]
                               [(if (:requires-grad? a)
                                  ((back-fn a) a (m/dot (ones-like @(:data b)) grad))
                                  a)]))))}))


(comment
  (let [out (broadcast-like (tape 5 true) (tape (dv [1 2 3])))
        grads ((:backward out) out (dv [1 2 3]))]
    grads))

(defn add [a b]
  (let [data (atom (m/add @(:data a) @(:data b)))]
    {:data data
     :id (uuid)
     :children [a b]
     :requires-grad? (or (:requires-grad? a) (:requires-grad? b))
     :backward (fn [{:keys [children] :as node} grad]
                 (-> node
                     (assoc :grad grad)
                     (update :children
                             (fn [cs]
                               (mapv (fn [a]
                                       (if (:requires-grad? a)
                                         ((back-fn a) a grad)
                                         a))
                                     cs)))))}))

;; TODO pytorch: 2nd order gradients with grad_fn
(comment
  (time
   (let [a (tape (dv [1 1 1]) true)
         b (tape (dv [1 1 1]) true)
         out (add a b)
         grads ((:backward out) out (dv [1 1 1]))]
     (-> grads :children first :grad)
     )))

(defn sub
  ([a]
   (sub (zeros-like a) a))
  ([a b]
   (let [data (atom (m/sub @(:data a) @(:data b)))]
     {:data data
      :id (uuid)
      :children [a b]
      :requires-grad? (or (:requires-grad? a) (:requires-grad? b))
      :backward (fn [{:keys [children] :as node} grad]
                  (-> node
                      (assoc :grad grad)
                      (update :children
                              (fn [[a b]]
                                [(if (:requires-grad? a)
                                   ((back-fn a) a grad)
                                   a)
                                 (if (:requires-grad? b)
                                   ((back-fn b) b (m/sub grad))
                                   b)]))))})))



(comment
  (let [out (sub (tape (dv 1 2 3))
                 (tape (dv 3 2 1) true))
        grads ((:backward out) out (dv 1 1 1))]
    grads))


(defn inner [a b]
  (let [data (atom (m/dot @(:data a) @(:data b)))]
    {:data data
     :id (uuid)
     :children [a b]
     :requires-grad? (or (:requires-grad? a) (:requires-grad? b))
     :backward (fn [{:keys [children] :as node} grad]
                 (-> node
                     (assoc :grad grad)
                     (update :children
                             (fn [[a b]]
                               [(if (:requires-grad? a)
                                  ((back-fn a) a (m/scale @(:data b) grad))
                                  a)
                                (if (:requires-grad? b)
                                  ((back-fn b) b (m/scale @(:data a) grad))
                                  b)]))))}))

(comment
  (let [x (dv [1 2 3])
        out (inner (tape x true) (tape x))
        grads ((back-fn out) out 2)]
    grads
    #_@(:data out)))



(defn gd! [grads]
  (let [{:keys [children data grad]} grads
        lr 0.01]
    (cond (not grad) nil
          (number? grad) (swap! data - (* grad lr))
          ;; TODO allow inplace
          :else (swap! data #(axpy (* -1.0 lr) grad %)))
    (if-not (empty? children)
      (let [new-children (mapv gd! children)]
        (assoc grads :children children))
      grads)))

(comment
  (let [a (tape (dv 1 1 1))
        b (tape (dv 1 2 3) true)
        out (inner a b)
        grads ((back-fn out) out 1)]
    (gd! grads)
    b))


(defn mul [m v]
  (let [data (atom (m/mmul @(:data m) @(:data v)))]
    {:data data
     :id (uuid)
     :children [m v]
     :requires-grad? (or (:requires-grad? m) (:requires-grad? v))
     :backward (fn [{:keys [children] :as node} grad]
                 (-> node
                     (assoc :grad grad)
                     (update :children
                             (fn [[m v]]
                               [(if (:requires-grad? m)
                                  ((back-fn m) m (m/outer-product grad @(:data v)))
                                  m)
                                (if (:requires-grad? v)
                                  ((back-fn v) v (m/mmul (trans @(:data m)) grad)) 
                                  v)]))))}))


(defn mmul [a b]
  (let [data (atom (m/mmul @(:data a) @(:data b)))]
    {:data data
     :id (uuid)
     :children [a b]
     :requires-grad? (or (:requires-grad? a) (:requires-grad? b))
     :backward (fn [{:keys [children] :as node} grad]
                 (-> node
                     (assoc :grad grad)
                     (update :children
                             (fn [[a b]]
                               [(if (:requires-grad? a)
                                  ((back-fn a) a (m/mmul grad (trans @(:data b))))
                                  a)
                                (if (:requires-grad? b)
                                  ((back-fn b) b (m/mmul (trans @(:data a)) grad)) 
                                  b)]))))}))



(comment

  (let [X (tape (trans (dge 1 2 [1 3])))
        Y (tape (dv [-10 -30]))
        c (tape 0 true)
        b (tape (dv [10]) true)]
    (loop [i 1000]
      (when (pos? i)
        (let [out (sub (add (mul X b) (broadcast-like c Y)) Y)
              out (inner out out)
              grads ((:backward out) out 1)]
          (when (zero? (mod i 100))
            (prn "Loss:" @(:data out) ", b:"  @(:data b) ", c:" @(:data c)))
          (gd! grads)
          (recur (dec i)))))
    b)


  (let [X (tape (trans (dge 1 10 (range 10))))
        Y (tape (dv (range 10)))
        c (tape 0 true)
        b (tape (dv [0]) true)]
    (loop [i 1000]
      (when (pos? i)
        (let [Y* (add (mul X b) (broadcast-like c Y))
              out (sub Y* Y)
              out (inner out out)
              grads ((:backward out) out 1)]
          (when (zero? (mod i 100))
            (prn "Loss:" @(:data out) ", b:"  @(:data b) " c:" @(:data c)))
          (gd! grads)
          (recur (dec i)))))
    b)

  (let [X (tape (trans (dge 2 3 [1 1 2 1 3 1])))
        Y (tape (dv [1 2 3]))
        c (tape 0 true)
        b (tape (dv [1 1]) true)]
    (loop [i 1000]
      (when (pos? i)
        (let [out (sub (add (mul X b) (broadcast-like c Y)) Y)
              out (inner out out)
              grads ((:backward out) out 1)]
          (when (zero? (mod i 100))
            (prn "Loss:" @(:data out) ", b:"  @(:data b) " c:" @(:data c)))
          (gd! grads)
          (recur (dec i)))))
    b)

  (let [X (tape (trans (dge 1 2 [1 5])))
        Y (tape (dv [2 10]))
        b (tape (dv [2]) true)
        z (sub (mul X b) Y)
        loss (inner z z)
        grads ((:backward loss) loss 1)]
    grads)) 


(defn sigmoid-g [x]
  (let [s (m/logistic x)]
    (m/mul s (m/sub 1.0 s))))


(defn sigmoid [x]
  (let [data (atom (m/logistic @(:data x)) #_(fmap esigmoid @(:data x)))]
    {:data data
     :id (uuid)
     :children [x]
     :requires-grad? (:requires-grad? x)
     :backward (fn [{:keys [children] :as node} grad]
                 (-> node
                     (assoc :grad grad)
                     (update :children
                             (fn [[x]]
                               [(if (:requires-grad? x)
                                  ((back-fn x) x (m/mul (sigmoid-g @(:data x)) grad))
                                  x)]))))}))

(comment
  (let [out (sigmoid (tape (dv [1 2 3]) true))
        grads ((:backward out) out (dv [1 1 1]))]
    grads))


(defn log-g [x]
  (m/div 1.0 x))

(comment
  (m/div (m/matrix [1 2 3])))

(defn log [x]
  (let [data (atom (m/log @(:data x)))]
    {:data data
     :id (uuid)
     :children [x]
     :requires-grad? (:requires-grad? x)
     :backward (fn [{:keys [children] :as node} grad]
                 (-> node
                     (assoc :grad grad)
                     (update :children
                             (fn [[x]]
                               [(if (:requires-grad? x)
                                  ((back-fn x) x (m/mul (log-g @(:data x)) grad))
                                  x)]))))}))

(comment
  (let [out (log (tape (dv [0.3 0.8 1.0]) true))
        grads ((:backward out) out (dv [1 1 1]))]
    out))

(defn emul [a b]
  (let [data (atom (m/mul @(:data a) @(:data b)))]
    {:data data
     :id (uuid)
     :children [a b]
     :requires-grad? (or (:requires-grad? a) (:requires-grad? b))
     :backward (fn [{:keys [children] :as node} grad]
                 (-> node
                     (assoc :grad grad)
                     (update :children
                             (fn [[a b]]
                               [(if (:requires-grad? a)
                                  ((back-fn a) a (m/mul @(:data b) grad))
                                  a)
                                (if (:requires-grad? b)
                                  ((back-fn b) b (m/mul @(:data a) grad))
                                  b)]))))}))

(defn bcrossent [a b]
  (inner (add (emul (log a) b)
              (emul (log (sub (ones-like a) a))
                    (sub (ones-like b) b)))
         (sub (ones-like a))))


(comment
 (let [out (bcrossent (tape (dv [0.5 0.5 0.5]) true)
                      (tape (dv [0 1 0])))
       grads ((:backward out) out 1)]
   grads))





(comment
  (let [a (tape (dv [1 -2 4]) true)
        b (tape (dv [1 1 1]))
        out (add a b)
        a-delta (dv [0.01 0.01 0.01])
        grads ((back-fn out) out a-delta)
        ;; numeric grad
        out-num (add (tape (dv [1.01 -2 4])) b)
        out-delta (sub out-num  out)
        ]
    (sum @(:data (sub (tape (:grad grads)) out-delta)))
    #_(broadcast-like 0.01 out-n)
    #_@(:data (sub out-n (emul out )))
    #_(:grad (first (:children grads)))))

(defn show-grad [n]
  (assoc n :backward (fn [a grad] (prn "grad: " grad) (assoc a :grad grad))))

(comment

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
            (prn "Loss:" @(:data out) ", b:"  @(:data b) @(:data c)))
          (gd! grads)
          (recur (dec i)))))
    b)

  )



(comment

  (defn rand-mat [m n]
    (dge m n (take (* m n) (repeatedly #(- (rand) 0.5)))))


  (let [x (tape (dge 3 2 [1 2 3 4 5 6]))
        in-dim 3
        h-dim 2
        w1 (tape (rand-mat h-dim in-dim) true)
        h1 (sigmoid (mmul w1 x))
        w2 (tape (rand-mat 1 h-dim) true)
        h2 (sigmoid (mmul w2 h1))
        _ (swap! (:data h2) row 0) ;; cast to vector
        loss (bcrossent h2 (tape (dv [1 0])))
        grads ((:backward loss) loss 1)]
    (gd! grads)))







