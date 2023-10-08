#!/usr/bin/env roseus
(require :robot-teleop-interface "package://eus_imitation/euslisp/robot-teleop-interface.l")

(defclass robot-vive-interface
  :super robot-teleop-interface
  :slots (vrbase-device-id
          larm-controller-device-id
          rarm-controller-device-id
          larm-controller-frame-id
          rarm-controller-frame-id
          trigger-wait-count
          menu-wait-count
          trackpad-wait-count
          grip-wait-count ;; TODO
          larm-trigger
          larm-prev-trigger
          larm-trigger-count
          larm-menu
          larm-prev-menu
          larm-menu-count
          larm-trackpad
          larm-prev-trackpad
          larm-trackpad-count
          larm-grip
          larm-prev-grip
          larm-grip-count
          rarm-trigger
          rarm-prev-trigger
          rarm-trigger-count
          rarm-menu
          rarm-prev-menu
          rarm-menu-count
          rarm-trackpad
          rarm-prev-trackpad
          rarm-trackpad-count
          rarm-grip
          rarm-prev-grip
          rarm-grip-count

          tfl
          vrbase-frame-id
          larm-tf-vrbase->controllerref
          larm-tf-base->gripperref
          larm-tf-controllerref->controller
          larm-tf-gripperref->gripper
          rarm-tf-vrbase->controllerref
          rarm-tf-base->gripperref
          rarm-tf-controllerref->controller
          rarm-tf-gripperref->gripper))

(defmethod robot-vive-interface
  (:init  (&rest args)
    (send-super* :init args)

   ;; for frame and vive controller
    (send self :set-val 'tfl (instance ros::transform-listener :init))
    (send self :set-val 'vrbase-frame-id (format nil "lighthouse_LHB_~A" "8E924CC3"))
    (send self :set-arm-val :larm :controller-device-id "FD35BD42")
    (send self :set-arm-val :rarm :controller-device-id "F7AFBF47")

    (send self :set-val 'recording nil)
    (send self :set-val 'trigger-wait-count 5)
    (send self :set-val 'menu-wait-count 5)
    (send self :set-val 'trackpad-wait-count 5)
    (send self :set-val 'grip-wait-count 5)
    self)
  (:ros-init ()
    (send-super :ros-init)
    (if larm-controller-device-id
      (progn
        (send self :set-arm-val :larm :controller-frame-id
              (format nil "controller_LHR_~A" larm-controller-device-id))
        (ros::subscribe (format nil "/~A/joy"
                                (send self :get-arm-val :larm :controller-frame-id))
                        sensor_msgs::Joy #'send self :controller-cb :larm)
        (ros::ros-info "Detected left arm controller device ID :  ~A" larm-controller-device-id))
      (ros::ros-warn "param: /eus_vive/l_controller_id is not set"))
    (if rarm-controller-device-id
      (progn
        (send self :set-arm-val :rarm :controller-frame-id
              (format nil "controller_LHR_~A" rarm-controller-device-id))
        (ros::subscribe (format nil "/~A/joy"
                                (send self :get-arm-val :rarm :controller-frame-id))
                        sensor_msgs::Joy #'send self :controller-cb :rarm)
        (ros::ros-info "Detected right arm controller device ID :  ~A" rarm-controller-device-id))
      (ros::ros-warn "param: /eus_vive/r_controller_id is not set"))
    t)
  (:reset-arm-val  (arm)
    (send-super :reset-arm-val arm)
    (send self :set-arm-val arm :trigger nil)
    (send self :set-arm-val arm :prev-trigger nil)
    (send self :set-arm-val arm :trigger-count 0)
    (send self :set-arm-val arm :menu nil)
    (send self :set-arm-val arm :prev-menu nil)
    (send self :set-arm-val arm :menu-count 0)
    (send self :set-arm-val arm :trackpad nil)
    (send self :set-arm-val arm :prev-trackpad nil)
    (send self :set-arm-val arm :trackpad-count 0)
    (send self :set-arm-val arm :grip nil)
    (send self :set-arm-val arm :prev-grip nil)
    (send self :set-arm-val arm :grip-count 0))
  (:calib-controller (arm)
    (let (tf-vrbase->controllerref tf-base->gripperref)
      (ros::ros-info "Calibrating controller ~A" arm)
      (send self :update-robot)
      (if (eq arm :arms) ;; when control both arms
          (progn
            (send self :set-arm-val :larm :tf-vrbase->controllerref (make-coords
                                              :pos (send (send self :get-vrbase->controller-coords :larm) :pos)
                                              :rot (send (send self :get-vrbase->controller-coords :larm) :rot)))
            (send self :set-arm-val :rarm :tf-base->gripperref (make-coords
                                                :pos (send (send self :get-arm-val :rarm :current-end-coords) :pos)
                                                :rot (send (send self :get-arm-val :rarm :current-end-coords) :rot))))
          (progn
            (send self :set-arm-val arm :tf-vrbase->controllerref (make-coords
                                              :pos (send (send self :get-vrbase->controller-coords arm) :pos)
                                              :rot (send (send self :get-vrbase->controller-coords arm) :rot)))
            (send self :set-arm-val arm :tf-base->gripperref (make-coords
                                                :pos (send (send self :get-arm-val arm :current-end-coords) :pos)
                                                :rot (send (send self :get-arm-val arm :current-end-coords) :rot)))))))
  (:get-vrbase->controller-coords (arm &key (wait-trigger nil) (wait-time 0.1))
    (let ((controller-frame-id (send self :get-arm-val arm :controller-frame-id))
          (start-time (ros::time 0))
          coords controller-trigger-button-p)
      (if (or (null vrbase-frame-id) (null controller-frame-id))
        (progn
          (ros::ros-warn "Frame ids are not set properly.")
          (ros::ros-warn "vrbase-frame-id: ~A" vrbase-frame-id)
          (ros::ros-warn "controller-frame-id: ~A" controller-frame-id)
          (return-from :get-vrbase->controller-coords nil)))
      (if wait-trigger
        (progn
          (ros::ros-info "Waiting for trigger: ~A~%" (string-downcase arm))
          (while (and (ros::ok) (not controller-trigger-button-p))
            (setq controller-trigger-button-p
                  (send self :controller-trigger-button-p arm :button-toggle t))
            (ros::spin-once)
            (ros::sleep))))
      (send tfl :wait-for-transform vrbase-frame-id controller-frame-id start-time wait-time)
      (setq coords (send tfl :lookup-transform vrbase-frame-id controller-frame-id start-time))
      coords))
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
          (send self :set-arm-val arm :target-end-rpy  (coerce (elt (rpy-angle (send target-coords :worldrot)) 0) float-vector))
      target-coords))

  (:controller-cb (arm msg) ;; NOTE no axes because it is not used
        (send self :set-arm-val arm :prev-trigger (send self :get-arm-val arm :trigger))
        (send self :set-arm-val arm :prev-menu (send self :get-arm-val arm :menu))
        (send self :set-arm-val arm :prev-trackpad (send self :get-arm-val arm :trackpad))
        (send self :set-arm-val arm :prev-grip (send self :get-arm-val arm :grip))
        (send self :set-arm-val arm :trigger (> (elt (send msg :buttons) 1) 0))
        (send self :set-arm-val arm :menu (> (elt (send msg :buttons) 0) 0))
        (send self :set-arm-val arm :trackpad (> (elt (send msg :buttons) 2) 0))
        (send self :set-arm-val arm :grip (> (elt (send msg :buttons) 3) 0)))
  (:controller-trigger-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          (if (eq arm :larm) 'larm-trigger 'rarm-trigger)
          :button-toggle button-toggle))
  (:controller-menu-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          (if (eq arm :larm) 'larm-menu 'rarm-menu)
          :button-toggle button-toggle))
  (:controller-trackpad-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          (if (eq arm :larm) 'larm-trackpad 'rarm-trackpad)
          :button-toggle button-toggle))
  (:controller-grip-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          (if (eq arm :larm) 'larm-grip 'rarm-grip)
          :button-toggle button-toggle))
  (:button-cb ()
    (send self :enable-cb :larm :mirror nil :button :trackpad) ;; check trackpad button to enable/disable arm
    (send self :enable-cb :rarm :mirror nil :button :trackpad) ;; check trackpad button to enable/disable arm
    (send self :gripper-cb :larm :wait nil :button :trigger) ;; check trigger button to open/close gripper
    (send self :gripper-cb :rarm :wait nil :button :trigger) ;; check trigger button to open/close gripper
    (send self :record-cb :button :menu) ;; check menu button to start/stop recording
    (send self :remove-cb :button :menu) ;; check menu button to remove rosbag recording
    (send self :init-cb :larm :button :grip) ;; check grip button to initialize robot
    (send self :init-cb :rarm :button :grip)))

(provide :robot-vive-interface)


