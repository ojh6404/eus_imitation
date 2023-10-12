#!/usr/bin/env roseus
(require :baxter-interface "package://baxtereus/baxter-interface.l")
(require :pr2-spacenav-interface "package://eus_imitation/euslisp/pr2-spacenav-interface.l")

(defun init-pr2-imitator (&key (imitation nil) (device :spacenav))
  (pr2-init)
  (objects (list *pr2*))
  (setq *pr2-imitator* (instance pr2-spacenav-interface :init :imitation imitation :device device))
  (send *pr2-imitator* :ros-init))
(defun main (&key (imitation nil) (device :spacenav))
  (init-pr2-imitator :imitation imitation :device device)
  (send *pr2-imitator* :main-loop :imitation imitation))
