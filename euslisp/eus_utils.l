#!/usr/bin/env roseus

(defun init-pr2-imitator ()
  (pr2-init)
  (objects (list *pr2*))
  (setq *pr2-imitator* (instance pr2-imitation-interface :init))
  (send *pr2-imitator* :ros-init))
(defun main (&key (arm :rarm) (imitation nil) (rot nil))
  (init-pr2-imitator)
  (send *pr2-imitator* :main-loop :arm arm :imitation imitation :rot rot))

;; print time consumed in function
(defun log-time (func &rest args)
  (let ((start-time (unix:gettimeofday))
        (end-time)
        (wait-time))

    (setf end-time (unix:gettimeofday))
    (apply func args)
    (setf wait-time (+ (* (- (elt end-time 0) (elt start-time 0)) 1000)
                        (/ (- (elt end-time 1) (elt start-time 1)) 1000.0)))
    (print wait-time)
)
