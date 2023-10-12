#!/usr/bin/env roseus
(require :fetch-interface "package://fetcheus/fetch-interface.l")
(require :fetch-spacenav-interface "package://eus_imitation/euslisp/fetch-spacenav-interface.l")

(defun init-fetch-imitator (&key (imitation nil) (device :spacenav))
  (fetch-init)
  (objects (list *fetch*))
  (setq *fetch-imitator* (instance fetch-spacenav-interface :init :imitation imitation :device device))
  (send *fetch-imitator* :ros-init))
(defun main (&key (imitation nil) (device :spacenav))
  (init-fetch-imitator :imitation imitation :device device)
  (send *fetch-imitator* :main-loop :imitation imitation))
