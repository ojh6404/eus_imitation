#!/usr/bin/env roseus
(require :fetch-interface "package://fetcheus/fetch-interface.l")
(require :fetch-imitation-interface "package://eus_imitation/euslisp/fetch-imitation-interface.l")



(defun init-fetch-imitator (&key (imitation nil) (device-type :vive))
  (fetch-init)
  (objects (list *fetch*))
  (setq *fetch-imitator* (instance fetch-imitation-interface :init :imitation imitation :device-type device-type))
  (send *fetch-imitator* :ros-init))
(defun main (&key (imitation nil) (device-type :vive))
  (init-fetch-imitator :imitation imitation :device-type device-type)
  (send *fetch-imitator* :main-loop :imitation imitation))
