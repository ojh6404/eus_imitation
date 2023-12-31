#!/usr/bin/env roseus

(defun init-pr2-imitator (&key (imitation nil) (device :vive))
  (require :pr2-interface "package://pr2eus/pr2-interface.l")
  (require :pr2-imitation-interface "package://eus_imitation/euslisp/pr2-imitation-interface.l")
  (pr2-init)
  (objects (list *pr2*))
  (setq *imitator* (instance pr2-imitation-interface :init :imitation imitation :device device))
  (send *imitator* :ros-init))

(defun init-fetch-imitator (&key (imitation nil) (device :vive))
  (require :fetch-interface "package://fetcheus/fetch-interface.l")
  (require :fetch-imitation-interface "package://eus_imitation/euslisp/fetch-imitation-interface.l")
  (fetch-init)
  (objects (list *fetch*))
  (setq *imitator* (instance fetch-imitation-interface :init :imitation imitation :device device))
  (send *imitator* :ros-init))

(defun init-baxter-imitator (&key (imitation nil) (device :vive))
  (require :baxter-interface "package://baxtereus/baxter-interface.l")
  (require :baxter-imitation-interface "package://eus_imitation/euslisp/baxter-imitation-interface.l")
  (baxter-init)
  (objects (list *baxter*))
  (setq *imitator* (instance baxter-imitation-interface :init :imitation imitation :device device))
  (send *imitator* :ros-init))

(defun init-kinova-imitator (&key (imitation nil) (device :spacenav))
  (require :kinovaeus "package://kinovaeus/kinova-interface.l")
  (require :kinova-imitation-interface "package://eus_imitation/euslisp/kinova-imitation-interface.l")
  (kinova-init :type :gen3_robotiq_2f_140)
  (objects (list *kinova*))
  (setq *imitator* (instance kinova-imitation-interface :init :imitation imitation :device device))
  (send *imitator* :ros-init))

(defun main (&key (robot :pr2) (imitation nil) (device :vive))
  (cond ((eq robot :pr2)
         (init-pr2-imitator :imitation imitation :device device))
        ((eq robot :fetch)
         (init-fetch-imitator :imitation imitation :device device))
        ((eq robot :baxter)
         (init-baxter-imitator :imitation imitation :device device))
        ((eq robot :kinova)
         (init-kinova-imitator :imitation imitation :device device))
        (t (format t "robot ~a is not supported." robot)))
  (send *imitator* :main-loop :imitation imitation))
