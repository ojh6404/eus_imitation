#!/usr/bin/env roseus
(require :pr2-interface "package://pr2eus/pr2-interface.l")
(require :pr2-vive-interface "package://eus_imitation/euslisp/pr2-vive-interface.l")

(defun init-pr2-imitator (&key (imitation nil) (device :vive))
  (pr2-init)
  (objects (list *pr2*))
  (setq *pr2-imitator* (instance pr2-vive-interface :init :imitation imitation :device device))
  (send *pr2-imitator* :ros-init))
(defun main (&key (imitation nil) (device :vive))
  (init-pr2-imitator :imitation imitation :device device)
  (send *pr2-imitator* :main-loop :imitation imitation))
