#!/usr/bin/env roseus
(require :robot-teleop-interface "package://eus_imitation/euslisp/robot-teleop-interface.l")

(defclass robot-spacenav-interface
  :super robot-teleop-interface
  :slots (spacenav-joy-topic-name
          spacenav-joy-axes-threshold

          larm-spacenav-joy-axes
          rarm-spacenav-joy-axes
          left-wait-count
          right-wait-count

          larm-left
          larm-prev-left
          larm-left-count
          larm-right
          larm-prev-right
          larm-right-count
          rarm-left
          rarm-prev-left
          rarm-left-count
          rarm-right
          rarm-prev-right
          rarm-right-count))



(defmethod robot-spacenav-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'spacenav-joy-topic-name "/spacenav/joy") ;; TODO only consider one spacenav now
    (send self :set-val 'left-wait-count 5)
    (send self :set-val 'right-wait-count 5)

    (send self :set-val 'spacenav-joy-axes-threshold 0.2)
    self)

  (:ros-init ()
    (send-super :ros-init)
    (ros::subscribe spacenav-joy-topic-name
                    sensor_msgs::Joy #'send self :controller-cb :rarm) ;; TODO currently only rarm
    t)
  (:reset-arm-val (arm)
    (send-super :reset-arm-val arm)
    (send self :set-arm-val arm :target-linear-velocity #f(0.0 0.0 0.0))
    (send self :set-arm-val arm :target-angular-velocity #f(0.0 0.0 0.0))
    (send self :set-arm-val arm :spacenav-joy-axes nil)
    (send self :set-arm-val arm :left nil)
    (send self :set-arm-val arm :prev-left nil)
    (send self :set-arm-val arm :left-count 0)
    (send self :set-arm-val arm :right nil)
    (send self :set-arm-val arm :prev-right nil)
    (send self :set-arm-val arm :right-count 0))
  (:controller-cb (arm msg)
    (send self :controller-axes-cb arm (send msg :axes))
    (send self :controller-buttons-cb arm (send msg :buttons)))
  (:controller-axes-cb (arm axes)
    (send self :set-arm-val arm :spacenav-joy-axes
      (mapcar #'(lambda (v) (if (> (abs v) spacenav-joy-axes-threshold) v 0))
              (coerce axes cons))))
  (:controller-buttons-cb (arm buttons) ;; NOTE no axes because it is not used
        (send self :set-arm-val arm :prev-left (send self :get-arm-val arm :left))
        (send self :set-arm-val arm :prev-right (send self :get-arm-val arm :right))
        (send self :set-arm-val arm :left (> (elt buttons 0) 0))
        (send self :set-arm-val arm :right (> (elt buttons 1) 0)))
  (:controller-left-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          (if (eq arm :larm) 'larm-left 'rarm-left)
          :button-toggle button-toggle))
  (:controller-right-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          (if (eq arm :larm) 'larm-right 'rarm-right)
          :button-toggle button-toggle))

  (:calib-controller (arm)
    ;; NOTE its dummy because we use joy-axes as target coords diff or joint velocity
    (ros::ros-info "Calibrating controller ~A" arm)
    (send self :update-robot))
  (:get-target-coords (arm &key (mirror nil) (pos-scale 100) (rot-scale 1))
    (let* ((spacenav-joy-axes (coerce (send self :get-arm-val arm :spacenav-joy-axes) float-vector))
           (joy-diff-pos (subseq spacenav-joy-axes 0 3))
           (joy-diff-rot (subseq spacenav-joy-axes 3 6))
           (target-diff-pos (scale pos-scale joy-diff-pos))
           (current-end-coords (send self :get-arm-val arm :current-end-coords))
           (current-end-pos (send current-end-coords :worldpos))
           (current-end-rot (send current-end-coords :worldrot))
           (target-pos (v+ current-end-pos target-diff-pos))
           (target-diff-rot
              (rpy-matrix
                (* rot-scale (elt joy-diff-rot 2))
                (* rot-scale (elt joy-diff-rot 1))
                (* rot-scale (elt joy-diff-rot 0))))
           (target-rot
              (if target-diff-rot
                  (send (send (make-coords :rot current-end-rot) :transform
                          (make-coords :rot target-diff-rot) :world)
                    :worldrot)
              current-end-rot))
           (target-coords (make-coords :pos target-pos :rot target-rot)))
          (send self :set-arm-val arm :target-coords target-coords)
          (send self :set-arm-val arm :target-end-pos (send target-coords :worldpos))
          (send self :set-arm-val arm :target-end-rpy (coerce (elt (rpy-angle (send target-coords :worldrot)) 0) float-vector))
        target-coords))

  (:get-target-cartesian-velocity (arm &key (linear-scale 0.1) (angular-scale 1.0))
    (let* ((target-linear-velocity (scale linear-scale (coerce (subseq (send self :get-arm-val arm :spacenav-joy-axes) 0 3) float-vector)))
           (target-angular-velocity (scale angular-scale (coerce (subseq (send self :get-arm-val arm :spacenav-joy-axes) 3 6) float-vector))))
      (send self :set-arm-val arm :target-linear-velocity target-linear-velocity)
      (send self :set-arm-val arm :target-angular-velocity target-angular-velocity)))

  (:button-cb ()
    ;; NOTE left-button : grasp, right-button : record, init, enable
    (send self :enable-cb :larm :mirror nil :button :right) ;; check trackpad button to enable/disable arm
    (send self :enable-cb :rarm :mirror nil :button :right) ;; check trackpad button to enable/disable arm
    (send self :gripper-cb :larm :wait nil :button :left) ;; check trigger button to open/close gripper
    (send self :gripper-cb :rarm :wait nil :button :left) ;; check trigger button to open/close gripper
    ;; (send self :init-cb :larm :button :right) ;; check grip button to initialize robot
    ;; (send self :init-cb :rarm :button :right)
    ;; (send self :record-cb :button :right)
    ;; (send self :remove-cb :button :right)
    ) ;; NOTE its dummy cause we have no larm controller

  (:loop-enable-arm (arm &key (mirror nil) (button :trackpad))
    (let (req res
            (enable-button-p
                (send self :controller-button-p
                    (if mirror (send self :get-opposite-arm arm) arm) button
                    :button-toggle t))
            (enable-p (send self :get-arm-val arm :enable))
            (record-p (send self :get-val 'recording)))
      (if enable-button-p
        (progn
          (if (and enable-p record-p)
            (progn
              (send robot-interface :cancel-angle-vector
                    :controller-type (send self :get-arm-controller arm))
              (if (find-method robot-interface :torso-controller)
                (send robot-interface :cancel-angle-vector :controller-type :torso-controller))
              (ros::ros-info "arm: ~A stopping" arm)
              (ros::ros-info "recording stop")
              (ros::wait-for-service record-trigger-service-name 1)
              (setq req (instance std_srvs::TriggerRequest :init))
              (setq res (ros::service-call record-trigger-service-name req))
              (send self :set-val 'recording nil)
              (ros::ros-info "Inititalizing robot...")
              (send self :init-pose)
              (send self :reset)
              (send self :update-robot)
              (ros::ros-info "Initiailized"))
            (progn
              (send self :update-robot)
              (ros::ros-info "recording start")
              (ros::wait-for-service record-trigger-service-name 1)
              (setq req (instance std_srvs::TriggerRequest :init))
              (setq res (ros::service-call record-trigger-service-name req nil))
              (send self :set-val 'recording t)
              (ros::ros-info "arm: ~A starting" arm)
              (send self :calib-controller arm)))
          (send self :set-arm-val arm :enable (not enable-p)))))))

(provide :robot-spacenav-interface)
