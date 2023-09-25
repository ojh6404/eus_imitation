#!/usr/bin/env roseus
(require :pr2-interface "package://pr2eus/pr2-interface.l")
(require :pr2-imitation-interface "package://eus_imitation/euslisp/pr2-imitation-interface.l")



(defun init-pr2-imitator (&key (imitation nil) (device-type :spacenav))
  (pr2-init)
  (objects (list *pr2*))
  (setq *pr2-imitator* (instance pr2-imitation-interface :init :imitation imitation :device-type device-type))
  (send *pr2-imitator* :ros-init))
(defun main (&key (imitation nil) (device-type :spacenav))
  (init-pr2-imitator :imitation imitation :device-type device-type)
  (send *pr2-imitator* :main-loop :imitation imitation))
