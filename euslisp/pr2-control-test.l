#!/usr/bin/env roseus
(require :pr2-interface "package://pr2eus/pr2-interface.l")
(require :pr2-spacenav-interface "package://eus_imitation/euslisp/pr2-spacenav-interface.l")

(defun main-loop-step (imitator)
    (send imitator :button-cb)
    (send imitator :update-robot)
    (send imitator :publish-robot-state)
    (send *irtviewer* :draw-objects)
    (send imitator :move-robot))
(defun main-loop (imitator &key (imitation nil) (rate 10))
  (send imitator :update-robot)
  (send imitator :init-pose)
  (if imitation ;; TODO
      (progn
        (ros::ros-info "Imitation control enabled")
        (ros::subscribe robot-action-topic-name eus_imitation::Float32MultiArrayStamped #'(lambda (msg) (send imitator :set-val 'robot-action (send msg :data))) 1)))
  (while (ros::ok)
    (ros::rate rate) ;; need to set every loop, otherwise it will be too fast. I don't know why.
    (main-loop-step imitator)
    (ros::spin-once)
    (ros::sleep)))


(defun init-pr2-imitator (&key (imitation nil) (device-type :vive))
  (pr2-init)
  (objects (list *pr2*))
  (setq *pr2-imitator* (instance pr2-spacenav-interface :init :imitation imitation :device-type device-type))
  (send *pr2-imitator* :ros-init))
(defun main (&key (imitation nil) (device-type :vive) (rate 10))
  (init-pr2-imitator :imitation imitation :device-type device-type)
  ;; (send *pr2-imitator* :main-loop :imitation imitation)
  (main-loop *pr2-imitator* :imitation imitation :rate rate))
