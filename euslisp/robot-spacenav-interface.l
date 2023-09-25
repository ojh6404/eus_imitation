#!/usr/bin/env roseus
(require :robot-teleop-interface "package://eus_imitation/euslisp/robot-teleop-interface.l")

(defclass robot-spacenav-interface
  :super robot-teleop-interface
  :slots (spacenav-joy-topic-name
          spacenav-joy-axes-threshold
          spacenav-trans-scale
          spacenav-rot-scale

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
          rarm-right-count


          z-translation-up
          z-translation-down))

(defmethod robot-spacenav-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'spacenav-joy-topic-name "/spacenav/joy") ;; TODO only consider one spacenav now
    (send self :set-val 'left-wait-count 5)
    (send self :set-val 'right-wait-count 5)

    (send self :set-val 'spacenav-trans-scale 50)
    (send self :set-val 'spacenav-rot-scale (/ pi 16.0))
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
  ;; (:get-target-coords (arm &key (pos-scale 1.5))
  ;;   (let* ((tf-vrbase->controller (send self :get-vrbase->controller-coords arm))
  ;;           (tf-base->gripperref (send self :get-arm-val arm :tf-base->gripperref))
  ;;           (tf-controllerref->controller (send (send (send self :get-arm-val arm :tf-vrbase->controllerref) :transformation tf-vrbase->controller) :copy-worldcoords))
  ;;           (tf-gripperref->gripper (make-coords
  ;;                                       :pos (scale pos-scale (send tf-controllerref->controller :pos))
  ;;                                       :rot (send tf-controllerref->controller :rot)))
  ;;           (target-coords (send tf-gripperref->gripper :transform (send tf-base->gripperref :copy-worldcoords) :world)))
  ;;         (send self :set-arm-val arm :target-coords target-coords)
  ;;         (send self :set-arm-val arm :target-end-pos (send target-coords :worldpos))
  ;;         (send self :set-arm-val arm :target-end-rpy  (elt (rpy-angle (send target-coords :worldrot)) 0))
  ;;     target-coords))

  (:get-target-coords (arm &key (mirror nil))
    ;; TODO: support dual arm control
    (send self :update-current-end-coords)
    (let* ((target-arm (if mirror (send self :get-opposite-arm arm) arm))
           (current-end-coords
             (send (send self :get-arm-val target-arm :current-end-coords)
                   :copy-worldcoords))
           (spacenav-joy-axes (send self :get-arm-val arm :spacenav-joy-axes))
           target-coords)
      (if (null (equal target-arm control-arm))
        (progn
          (ros::ros-error "Unsupported control arm: ~A" target-arm)
          (return-from :get-target-coords nil)))
      (if (some #'(lambda (x) (> (abs x) 0)) spacenav-joy-axes)
        (let* ((current-end-coords-pos (send current-end-coords :worldpos))
               (current-end-coords-rot (send current-end-coords :worldrot))
               (update-pos
                 (or (> (abs (elt spacenav-joy-axes 0)) 0)
                     (> (abs (elt spacenav-joy-axes 1)) 0)
                     (> (abs (elt spacenav-joy-axes 2)) 0)))
               (diff-target-pos
                 (if update-pos
                   (float-vector
                     (* scale spacenav-trans-scale (elt spacenav-joy-axes 0))
                     (* scale spacenav-trans-scale (elt spacenav-joy-axes 1))
                     (* scale spacenav-trans-scale (elt spacenav-joy-axes 2)))))
               (target-pos
                 (if diff-target-pos
                   (v+ current-end-coords-pos diff-target-pos)
                   current-end-coords-pos))
               (update-rot
                 (and (null update-pos)
                      (or (> (abs (elt spacenav-joy-axes 3)) 0)
                          (> (abs (elt spacenav-joy-axes 4)) 0)
                          (> (abs (elt spacenav-joy-axes 5)) 0))))
               (diff-target-rot
                 (if update-rot
                   (rpy-matrix
                     (* spacenav-rot-scale (elt spacenav-joy-axes 5))
                     (* spacenav-rot-scale (elt spacenav-joy-axes 4))
                     (* spacenav-rot-scale (elt spacenav-joy-axes 3)))))
               (target-rot
                 (if diff-target-rot
                   (send (send (make-coords :rot current-end-coords-rot) :transform
                               (make-coords :rot diff-target-rot) :world)
                         :worldrot)
                   current-end-coords-rot)))
          (if mirror
            (progn
              (setq target-pos (send self :get-mirror-pos target-pos))
              (setq target-rot (send self :get-mirror-rot target-rot))))
          (setq target-coords
                (make-coords :pos target-pos
                             :rot target-rot))))
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
          (send self :set-arm-val arm :enable (not enable-p))))))
)

(provide :robot-spacenav-interface)
