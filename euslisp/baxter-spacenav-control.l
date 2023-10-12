#!/usr/bin/env roseus
(require :baxter-interface "package://baxtereus/baxter-interface.l")
(require :baxter-spacenav-interface "package://eus_imitation/euslisp/baxter-spacenav-interface.l")

(defun init-baxter-imitator (&key (imitation nil) (device :spacenav))
  (baxter-init :safe nil :moveit nil)
  (objects (list *baxter*))
  (setq *imitator* (instance baxter-spacenav-interface :init :imitation imitation :device device))
  (send *imitator* :ros-init))
(defun main (&key (imitation nil) (device :spacenav))
  (init-baxter-imitator :imitation imitation :device device)
  (send *imitator* :main-loop :imitation imitation))
