#!/usr/bin/env roseus
(require :fetch-interface "package://fetcheus/fetch-interface.l")
(require :fetch-vive-interface "package://eus_imitation/euslisp/fetch-vive-interface.l")

(defun init-fetch-imitator (&key (imitation nil) (device :vive))
  (fetch-init)
  (objects (list *fetch*))
  (setq *imitator* (instance fetch-vive-interface :init :imitation imitation :device device))
  (send *imitator* :ros-init))
(defun main (&key (imitation nil) (device :vive))
  (init-fetch-imitator :imitation imitation :device device)
  (send *imitator* :main-loop :imitation imitation))
