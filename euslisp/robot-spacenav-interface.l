#!/usr/bin/env roseus
(require :robot-teleop-interface "package://eus_imitation/euslisp/robot-teleop-interface.l")

(defclass robot-spacenav-interface
  :super robot-teleop-interface
  :slots (spacenav-joy-topic-name
          spacenav-joy-axes
          spacenav-joy-axes-threshold
          spacenav-joy-buttons
          spacenav-trans-scale
          spacenav-rot-scale

          left-button
          left-button-count
          left-button-wait-count
          prev-left-button
          right-button
          right-button-count
          right-button-wait-count
          prev-right-button
          z-translation-up
          z-translation-down))

(defmethod robot-spacenav-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'spacenav-joy-topic-name "/spacenav/joy")
    (send self :set-val 'left-button nil)
    (send self :set-val 'left-button-count 0)
    (send self :set-val 'prev-left-button nil)
    (send self :set-val 'right-button nil)
    (send self :set-val 'right-button-count 0)
    (send self :set-val 'prev-right-button nil)
    (send self :set-val 'spacenav-trans-scale 50)
    (send self :set-val 'spacenav-rot-scale (/ pi 16.0))
    (send self :set-val 'spacenav-joy-axes-threshold 0.2)
    self)
  (:ros-init ()
    (send-super :ros-init)
    (ros::subscribe spacenav-joy-topic-name
                    sensor_msgs::Joy #'send self :controller-cb) ;; currently only rarm
    t)
  (:controller-cb (msg)
    (send self :controller-axes-cb (send msg :axes))
    (send self :controller-buttons-cb (send msg :buttons)))
  (:controller-axes-cb (axes)
    (send self :set-val 'spacenav-joy-axes
      (mapcar #'(lambda (v) (if (> (abs v) spacenav-joy-axes-threshold) v 0))
              (coerce axes cons))))
  (:controller-buttons-cb (buttons)
    (send self :set-val 'prev-left-button left-button)
    (send self :set-val 'prev-right-button right-button)
    (send self :set-val 'left-button (> (elt buttons 0) 0))
    (send self :set-val 'right-button (> (elt buttons 1) 0)))
  (:calib-controller (arm)
    (ros::ros-info "Calibrating controller ~A" arm)
    (send self :update-robot))
  (:get-target-coords (arm &key (pos-scale 1.5))
    (let* ((tf-vrbase->controller (send self :get-vrbase->controller-coords arm))
            (tf-base->gripperref (send self :get-arm-val arm :tf-base->gripperref))
            (tf-controllerref->controller (send (send (send self :get-arm-val arm :tf-vrbase->controllerref) :transformation tf-vrbase->controller) :copy-worldcoords))
            (tf-gripperref->gripper (make-coords
                                        :pos (scale pos-scale (send tf-controllerref->controller :pos))
                                        :rot (send tf-controllerref->controller :rot)))
            (target-coords (send tf-gripperref->gripper :transform (send tf-base->gripperref :copy-worldcoords) :world)))
          (send self :set-arm-val arm :target-coords target-coords)
          (send self :set-arm-val arm :target-end-pos (send target-coords :worldpos))
          (send self :set-arm-val arm :target-end-rpy  (elt (rpy-angle (send target-coords :worldrot)) 0))
      target-coords))

  (:get-target-cartesion-joint-vel (arm &key (vel-scale 0.5))
                                   (print "not implemented yet"))

  (:controller-left-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          'left-button :button-toggle button-toggle))
  (:controller-right-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          'right-button :button-toggle button-toggle))


  ;; (:button-cb ()
  ;;   ;; NOTE left-button : grasp, right-button : record, init, enable
  ;;   (send self :enable-cb :larm :mirror nil :button :trackpad) ;; check trackpad button to enable/disable arm
  ;;   (send self :enable-cb :rarm :mirror nil :button :trackpad) ;; check trackpad button to enable/disable arm
  ;;   (send self :gripper-cb :larm :wait nil :button :trigger) ;; check trigger button to open/close gripper
  ;;   (send self :gripper-cb :rarm :wait nil :button :trigger) ;; check trigger button to open/close gripper
  ;;   (send self :record-cb :button :menu) ;; check menu button to start/stop recording
  ;;   (send self :remove-cb :button :menu) ;; check menu button to remove rosbag recording
  ;;   (send self :init-cb :larm :button :grip) ;; check grip button to initialize robot
  ;;   (send self :init-cb :rarm :button :grip))

)

(provide :robot-spacenav-interface)


(defun test(&key (imitation nil) (device-type :spacenav))
  (setq *ri* 1)
  (setq *ti* (instance robot-spacenav-interface :init :imitation imitation :device-type device-type))
  (send *ti* :ros-init)
  (while (ros::ok)
    (print (send *ti* :get-val 'left-button))
    (print (send *ti* :get-val 'spacenav-joy-axes))
    (ros::spin-once)
    (ros::sleep))
)
