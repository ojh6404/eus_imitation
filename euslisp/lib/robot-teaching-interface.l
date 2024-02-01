#!/usr/bin/env roseus
(ros::roseus-add-msgs "eus_imitation")
(ros::load-ros-manifest "roseus")
(ros::roseus-add-msgs "sensor_msgs")
(ros::roseus "eus_imitation" :anonymous t)

(ros::set-logger-level "ros.roseus" 5)
(ros::load-ros-manifest "actionlib_msgs")
(ros::load-ros-manifest "moveit_msgs")
(ros::load-ros-manifest "sensor_msgs")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "std_srvs")

(defclass robot-teaching-interface
  :super propertied-object
  :slots (;; robot control
          device
          rate
          tfl
          control-arm
          larm-enable
          rarm-enable
          torse-enable-p ;; TODO
          recording
          loop-enable
          larm-loop-enable
          rarm-loop-enable
          larm-enable-p
          rarm-enable-p
          data-collection
          gripper-threshold
          pos-scale
          rot-scale
          mirror

          ;; vive controller
          vrbase-device-id
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
          vrbase-frame-id
          larm-tf-vrbase->controllerref
          larm-tf-base->gripperref
          larm-tf-controllerref->controller
          larm-tf-gripperref->gripper
          rarm-tf-vrbase->controllerref
          rarm-tf-base->gripperref
          rarm-tf-controllerref->controller
          rarm-tf-gripperref->gripper

          ;; spacenav controller
          spacenav-joy-topic-name
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
          rarm-right-count

          ;; rogik controller
          larm-rogik-device-id
          rarm-rogik-device-id
          larm-rogik-frame-id
          rarm-rogik-frame-id
          blue-wait-count
          red-wait-count
          display-wait-count
          larm-blue
          larm-prev-blue
          larm-blue-count
          larm-red
          larm-prev-red
          larm-red-count
          larm-display
          larm-prev-display
          larm-display-count
          rarm-blue
          rarm-prev-blue
          rarm-blue-count
          rarm-red
          rarm-prev-red
          rarm-red-count
          rarm-display
          rarm-prev-display
          rarm-display-count
          base-frame-id
          larm-tf-base->rogik
          rarm-tf-base->rogik))

(defmethod robot-teaching-interface
  (:init  (&key (rate 10) (imitation nil) (device :vive))
    (send self :set-val 'data-collection (not imitation)) ;; mode
    (send self :set-val 'device device)
    (send self :set-val 'rate rate)
    (send self :set-val 'mirror nil)
    (send self :controller-method device :init)
    self)
  (:ros-init ()
    (send self :controller-method device :ros-init)
    t)
  (:controller-method (device method-name &rest args)
    (let* ((device-method
            (read-from-string (format nil ":~A-~A" (string-downcase device) (string-downcase method-name)))))
      (if (find-method self device-method)
        (send* self device-method args)
        (progn
          (ros::ros-error (format nil "cannot find device method: ~A" device-method))
          nil))))
  (:get-arm-val (arm name)
    (if (not (or (eq arm :arms) (eq arm :larm) (eq arm :rarm)))
      (progn
        (ros::ros-error (format nil "Invalid args: (send self :get-arm-val ~A ~A)" arm name))
        (return-from :get-arm-val nil)))
    (let ((valname (format nil "~A-~A" (string-upcase arm) (string-upcase name))))
      (if (eq arm :arms)
        (progn
          (if (and (eq (send self :get-arm-val :larm name) nil)  (eq (send self :get-arm-val :rarm name) nil))
            (return-from :get-arm-val nil)
            (progn
              (return-from :get-arm-val (list (send self :get-arm-val :larm name)
                                              (send self :get-arm-val :rarm name)))))))
      (if (find (read-from-string valname) (mapcar #'car (cdr (send self :slots))))
        (send self :get-val (intern valname)))))
  (:set-arm-val (arm name val)
    (if (not (or (eq arm :arms) (eq arm :larm) (eq arm :rarm)))
      (progn
        (ros::ros-error (format nil "Invalid args: (send self :set-arm-val: ~A ~A ~A)" arm name val))
        (return-from :set-arm-val nil)))
    (let ((valname (format nil "~A-~A" (string-upcase arm) (string-upcase name))))
      (if (eq arm :arms)
        (progn
          (if (listp val)
            (progn
              (send self :set-arm-val :larm name (first val))
              (send self :set-arm-val :rarm name (second val)))
            (progn
              (send self :set-arm-val :larm name val)
              (send self :set-arm-val :rarm name val))))
        (send self :set-val (intern valname) val))))
  (:reset ()
    (send self :reset-arm-val :larm)
    (send self :reset-arm-val :rarm))
  (:reset-arm-val (arm)
    (send self :controller-method device :reset-arm-val arm))
  (:calib-controller  (arm)
    (send self :controller-method device :calib-controller arm))
  (:get-target-coords (arm &key (pos-scale 100) (rot-scale (/ pi 4)))
    (send self :controller-method device :get-target-coords arm :pos-scale pos-scale :rot-scale rot-scale))
  (:controller-cb (arm msg) ;; NOTE no axes because it is not used
    (send self :controller-method device :controller-cb arm msg))
  (:button-cb ()
    (send self :controller-method device :button-cb))
  (:loop-enable-arm (arm &key (mirror nil) (button :trackpad))
    (send self :controller-method device :loop-enable-arm arm :mirror mirror :button button))

  ;; controller common
  (:enable-cb (arm &key (mirror nil) (button :trackpad))
    (send self :loop-enable-arm arm :mirror mirror :button button)
    (if (and (send self :get-arm-val :larm :enable) (send self :get-arm-val :rarm :enable))
      (send self :set-val 'control-arm :arms)
      (if (send self :get-arm-val :larm :enable)
        (send self :set-val 'control-arm :larm)
        (if (send self :get-arm-val :rarm :enable)
          (send self :set-val 'control-arm :rarm)
          (send self :set-val 'control-arm nil)))))
  (:record-cb (&key (button :menu))
    (let (req res (record-button-p
            (send self :controller-button-p :rarm button :button-toggle t))
          (record-p (send self :get-val 'recording)))
      (if record-button-p
          (progn
            (if record-p
              (progn
                ;; (unix:usleep (* 1000 100))
                (ros::ros-info "recording stop")
                (ros::wait-for-service record-trigger-service-name 1)
                (setq req (instance std_srvs::TriggerRequest :init))
                (setq res (ros::service-call record-trigger-service-name req))
                (send self :set-val 'recording nil))
              (progn
                ;; (unix:usleep (* 1000 1000))
                (ros::ros-info "recording start")
                (ros::wait-for-service record-trigger-service-name 1)
                (setq req (instance std_srvs::TriggerRequest :init))
                (setq res (ros::service-call record-trigger-service-name req nil))
                (send self :set-val 'recording t)))
            (send self :set-val 'recording (not record-p))))))
  (:remove-cb (&key (button :menu))
    (let (req res (remove-button-p
            (send self :controller-button-p :larm button :button-toggle t)))
      (if remove-button-p
          (progn
            (ros::wait-for-service remove-trigger-service-name 1)
            (setq req (instance std_srvs::TriggerRequest :init))
            (setq res (ros::service-call remove-trigger-service-name req))))))
  (:init-cb (arm &key (button :grip))
    (let ((init-button-p
            (send self :controller-button-p arm button :button-toggle t)))
      (if init-button-p
          (progn
            (ros::ros-info "Inititalizing robot...")
            (send self :init-pose)
            (send self :reset)
            (send self :update-robot)
            (ros::ros-info "Initiailized")))))
  (:gripper-cb (arm &key (wait nil) (button :trigger)) ;; TODO, when 2 button, open/close gripper. do nothing when buttons are not pressed
    (if (send self :get-arm-val arm button) ;; NOTE controller-button-p is not used cause it is not toggle
        (send self :set-arm-val arm :grasp 1) ;;  we can set this t or nil but in case of use of this variable as state
        (send self :set-arm-val arm :grasp 0)))
  (:controller-button-p (arm button &key (button-toggle t))
    (let ((controller-button-method
            (read-from-string (format nil ":controller-~A-button-p" (string-downcase button)))))
      (if (find-method self controller-button-method)
        (send self controller-button-method arm :button-toggle button-toggle)
        (progn
          (ros::ros-error (format nil "cannot find method: ~A" controller-button-method))
          nil))))
  (:controller-button-common-p (button &key (button-toggle t))
    (let* ((button-prefix (subseq (format nil "~A" button) 0 5))
           (prev-button
             (if (or (equal button-prefix "larm-")
                     (equal button-prefix "rarm-"))
               (read-from-string
                 (format nil "~Aprev-~A" button-prefix (subseq (format nil "~A" button) 5)))
               (read-from-string (format nil "prev-~A" button))))
           (button-counter (read-from-string (format nil "~A-count" button)))
           (button-wait-counter
             (read-from-string
               (format nil "~A-wait-count"
                       (if (or (equal button-prefix "larm-")
                               (equal button-prefix "rarm-"))
                         (subseq (format nil "~A" button) 5)
                         button))))
           (button-p (send self :get-val button))
           (prev-button-p (send self :get-val prev-button))
           (button-count (send self :get-val button-counter))
           (button-wait-count (send self :get-val button-wait-counter))
           (controller-button-p (and (not (eq button-p prev-button-p))
                                     (if button-toggle button-p t))))
      (if button-toggle
        (if (> button-count 0)
          (progn
            (send self :set-val button-counter (- button-count 1))
            nil)
          (progn
            (if controller-button-p
              (progn
                (send self :set-val button-counter button-wait-count)
                (cons controller-button-p button-p)))))
        (if controller-button-p (cons controller-button-p button-p)))))

  ;; for vive controller
  (:vive-init ()
    (send self :set-val 'pos-scale 1.0)
    (send self :set-val 'rot-scale (/ pi 4))
    (send self :set-val 'tfl (instance ros::transform-listener :init))
    (send self :set-val 'vrbase-frame-id
     (format nil "lighthouse_LHB_~A" (ros::get-param "/eus_imitation/b_lighthouse_id")))
    (send self :set-arm-val :larm :controller-device-id
     (ros::get-param "/eus_imitation/l_controller_id"))
    (send self :set-arm-val :rarm :controller-device-id
     (ros::get-param "/eus_imitation/r_controller_id"))
    (send self :set-val 'recording nil)
    (send self :set-val 'trigger-wait-count 5)
    (send self :set-val 'menu-wait-count 5)
    (send self :set-val 'trackpad-wait-count 5)
    (send self :set-val 'grip-wait-count 5))
  (:vive-ros-init ()
      (if larm-controller-device-id
        (progn
          (send self :set-arm-val :larm :controller-frame-id
                (format nil "controller_LHR_~A" larm-controller-device-id))
          (ros::subscribe (format nil "/~A/joy"
                                  (send self :get-arm-val :larm :controller-frame-id))
                          sensor_msgs::Joy #'send self :vive-controller-cb :larm)
          (ros::ros-info "Detected left arm controller device ID :  ~A" larm-controller-device-id))
        (ros::ros-warn "param: /eus_imitation/l_controller_id is not set"))
      (if rarm-controller-device-id
        (progn
          (send self :set-arm-val :rarm :controller-frame-id
                (format nil "controller_LHR_~A" rarm-controller-device-id))
          (ros::subscribe (format nil "/~A/joy"
                                  (send self :get-arm-val :rarm :controller-frame-id))
                          sensor_msgs::Joy #'send self :vive-controller-cb :rarm)
          (ros::ros-info "Detected right arm controller device ID :  ~A" rarm-controller-device-id))
        (ros::ros-warn "param: /eus_imitation/r_controller_id is not set"))
    t)
  (:vive-reset-arm-val (arm)
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
  (:vive-calib-controller (arm)
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
                                                :rot (send (send self :get-arm-val :rarm :current-end-coords) :rot)))
            (send self :set-arm-val :rarm :tf-vrbase->controllerref (make-coords
                                              :pos (send (send self :get-vrbase->controller-coords :rarm) :pos)
                                              :rot (send (send self :get-vrbase->controller-coords :rarm) :rot)))
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
  (:vive-get-target-coords (arm &key (pos-scale 1.5) (rot-scale (/ pi 4)))
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
  (:vive-controller-cb (arm msg) ;; NOTE no axes because it is not used
        (send self :set-arm-val arm :prev-trigger (send self :get-arm-val arm :trigger))
        (send self :set-arm-val arm :prev-menu (send self :get-arm-val arm :menu))
        (send self :set-arm-val arm :prev-trackpad (send self :get-arm-val arm :trackpad))
        (send self :set-arm-val arm :prev-grip (send self :get-arm-val arm :grip))
        (send self :set-arm-val arm :trigger (> (elt (send msg :buttons) 1) 0))
        (send self :set-arm-val arm :menu (> (elt (send msg :buttons) 0) 0))
        (send self :set-arm-val arm :trackpad (> (elt (send msg :buttons) 2) 0))
        (send self :set-arm-val arm :grip (> (elt (send msg :buttons) 3) 0)))
  (:vive-button-cb ()
    (send self :enable-cb :larm :mirror nil :button :trackpad) ;; check trackpad button to enable/disable arm
    (send self :enable-cb :rarm :mirror nil :button :trackpad) ;; check trackpad button to enable/disable arm
    (send self :gripper-cb :larm :wait nil :button :trigger) ;; check trigger button to open/close gripper
    (send self :gripper-cb :rarm :wait nil :button :trigger) ;; check trigger button to open/close gripper
    (send self :record-cb :button :menu) ;; check menu button to start/stop recording
    (send self :remove-cb :button :menu) ;; check menu button to remove rosbag recording
    (send self :init-cb :larm :button :grip) ;; check grip button to initialize robot
    (send self :init-cb :rarm :button :grip))
  (:vive-loop-enable-arm (arm &key (mirror nil) (button :trackpad))
    (let ((enable-button-p
            (send self :controller-button-p
                  (if mirror (send self :get-opposite-arm arm) arm) button
                  :button-toggle t))
          (enable-p (send self :get-arm-val arm :enable)))
      (if enable-button-p
        (progn
          (if enable-p
            (progn
              (send robot-interface :cancel-angle-vector
                    :controller-type (send self :get-arm-controller arm))
              (if (find-method robot-interface :torso-controller)
                (send robot-interface :cancel-angle-vector :controller-type :torso-controller))
              (ros::ros-info "arm: ~A stopping" arm))
            (progn
              (send self :update-robot)
              (ros::ros-info "arm: ~A starting" arm)
              (send self :calib-controller arm)))
          (send self :set-arm-val arm :enable (not enable-p))))))
  (:get-vrbase->controller-coords (arm &key (wait-time 0.1))
    (let ((controller-frame-id (send self :get-arm-val arm :controller-frame-id))
          (start-time (ros::time 0))
          coords)
      (if (or (null vrbase-frame-id) (null controller-frame-id))
        (progn
          (ros::ros-warn "Frame ids are not set properly.")
          (ros::ros-warn "vrbase-frame-id: ~A" vrbase-frame-id)
          (ros::ros-warn "controller-frame-id: ~A" controller-frame-id)
          (return-from :get-vrbase->controller-coords nil)))
      (send tfl :wait-for-transform vrbase-frame-id controller-frame-id start-time wait-time)
      (setq coords (send tfl :lookup-transform vrbase-frame-id controller-frame-id start-time))
      coords))
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

  ;; for spacenav controller
  (:spacenav-init ()
    (send self :set-val 'pos-scale 100)
    (send self :set-val 'rot-scale (/ pi 4))
    (send self :set-val 'spacenav-joy-topic-name "/spacenav/joy") ;; TODO only consider one spacenav now
    (send self :set-val 'left-wait-count 5)
    (send self :set-val 'right-wait-count 5)
    (send self :set-val 'spacenav-joy-axes-threshold 0.2))
  (:spacenav-ros-init ()
    (ros::subscribe spacenav-joy-topic-name
                    sensor_msgs::Joy #'send self :spacenav-controller-cb :rarm) ;; TODO currently only rarm
    t)
  (:spacenav-reset-arm-val (arm)
    (send self :set-arm-val arm :target-linear-velocity #f(0.0 0.0 0.0))
    (send self :set-arm-val arm :target-angular-velocity #f(0.0 0.0 0.0))
    (send self :set-arm-val arm :spacenav-joy-axes nil)
    (send self :set-arm-val arm :left nil)
    (send self :set-arm-val arm :prev-left nil)
    (send self :set-arm-val arm :left-count 0)
    (send self :set-arm-val arm :right nil)
    (send self :set-arm-val arm :prev-right nil)
    (send self :set-arm-val arm :right-count 0))
  (:spacenav-calib-controller (arm)
    (ros::ros-info "Calibrating controller ~A" arm)
    (send self :update-robot))
  (:spacenav-get-target-coords (arm &key (pos-scale 100) (rot-scale (/ pi 4)))
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
  (:spacenav-controller-cb (arm msg)
    (send self :spacenav-controller-axes-cb arm (send msg :axes))
    (send self :spacenav-controller-buttons-cb arm (send msg :buttons)))
  (:spacenav-controller-axes-cb (arm axes)
    (send self :set-arm-val arm :spacenav-joy-axes
      (mapcar #'(lambda (v) (if (> (abs v) spacenav-joy-axes-threshold) v 0))
              (coerce axes cons))))
  (:spacenav-button-cb ()
    (send self :enable-cb :larm :mirror nil :button :right) ;; check trackpad button to enable/disable arm
    (send self :enable-cb :rarm :mirror nil :button :right) ;; check trackpad button to enable/disable arm
    (send self :gripper-cb :larm :wait nil :button :left) ;; check trigger button to open/close gripper
    (send self :gripper-cb :rarm :wait nil :button :left))
  (:spacenav-controller-buttons-cb (arm buttons) ;; NOTE no axes because it is not used
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
  (:spacenav-get-target-cartesian-velocity (arm &key (linear-scale 0.1) (angular-scale 1.0))
    (let* ((target-linear-velocity (scale linear-scale (coerce (subseq (send self :get-arm-val arm :spacenav-joy-axes) 0 3) float-vector)))
           (target-angular-velocity (scale angular-scale (coerce (subseq (send self :get-arm-val arm :spacenav-joy-axes) 3 6) float-vector))))
      (send self :set-arm-val arm :target-linear-velocity target-linear-velocity)
      (send self :set-arm-val arm :target-angular-velocity target-angular-velocity)))
  (:spacenav-loop-enable-arm (arm &key (mirror nil) (button :right))
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

  ;; rogik controller
  (:rogik-init ()
    (send self :set-val 'pos-scale 1.0)
    (send self :set-val 'rot-scale 1.0)
    (send self :set-val 'tfl (instance ros::transform-listener :init))
    (send self :set-val 'base-frame-id "base_footprint")
    (send self :set-arm-val :larm :rogik-device-id
     (ros::get-param "/eus_imitation/l_rogik_id"))
    (send self :set-arm-val :rarm :rogik-device-id
     (ros::get-param "/eus_imitation/r_rogik_id"))
    (send self :set-val 'recording nil)
    (send self :set-val 'blue-wait-count 5)
    (send self :set-val 'red-wait-count 5)
    (send self :set-val 'display-wait-count 5))
  (:rogik-ros-init ()
      (if larm-rogik-device-id
        (progn
          (send self :set-arm-val :larm :rogik-frame-id
                (format nil "rogik_~A_frame" larm-rogik-device-id))
          (ros::subscribe (format nil "/rogik_~A/joy"
                                  (send self :get-arm-val :larm :rogik-device-id))
                          sensor_msgs::Joy #'send self :rogik-controller-cb :larm)
          (ros::ros-info "Detected left arm rogik device ID :  ~A" larm-rogik-device-id))
        (ros::ros-warn "param: /eus_imitation/l_rogik_id is not set"))
      (if rarm-rogik-device-id
        (progn
          (send self :set-arm-val :rarm :rogik-frame-id
                (format nil "rogik_~A_frame" rarm-rogik-device-id))
          (ros::subscribe (format nil "/rogik_~A/joy"
                                  (send self :get-arm-val :rarm :rogik-device-id))
                          sensor_msgs::Joy #'send self :rogik-controller-cb :rarm)
          (ros::ros-info "Detected right arm rogik device ID :  ~A" rarm-rogik-device-id))
        (ros::ros-warn "param: /eus_imitation/r_rogik_id is not set"))
    t)
  (:rogik-reset-arm-val (arm)
    (send self :set-arm-val arm :blue nil)
    (send self :set-arm-val arm :prev-blue nil)
    (send self :set-arm-val arm :blue-count 0)
    (send self :set-arm-val arm :red nil)
    (send self :set-arm-val arm :prev-red nil)
    (send self :set-arm-val arm :red-count 0)
    (send self :set-arm-val arm :display nil)
    (send self :set-arm-val arm :prev-display nil)
    (send self :set-arm-val arm :display-count 0))
  (:rogik-calib-controller (arm)
    (ros::ros-info "Calibrating controller ~A" arm)
    (send self :update-robot))
  (:rogik-get-target-coords (arm &key (pos-scale 1.0) (rot-scale 1.0))
    (let* ((target-coords (send self :get-base->gripper-coords arm)))
          (send self :set-arm-val arm :target-coords target-coords)
          (send self :set-arm-val arm :target-end-pos (send target-coords :worldpos))
          (send self :set-arm-val arm :target-end-rpy  (coerce (elt (rpy-angle (send target-coords :worldrot)) 0) float-vector))
      target-coords))
  (:rogik-controller-cb (arm msg) ;; NOTE no axes because it is not used
        (send self :set-arm-val arm :prev-blue (send self :get-arm-val arm :blue))
        (send self :set-arm-val arm :prev-red (send self :get-arm-val arm :red))
        (send self :set-arm-val arm :prev-display (send self :get-arm-val arm :display))
        (send self :set-arm-val arm :blue (> (elt (send msg :buttons) 0) 0))
        (send self :set-arm-val arm :red (> (elt (send msg :buttons) 1) 0))
        (send self :set-arm-val arm :display (> (elt (send msg :buttons) 2) 0)))
  (:rogik-button-cb ()
    (send self :enable-cb :larm :mirror nil :button :display) ;; check trackpad button to enable/disable arm
    (send self :enable-cb :rarm :mirror nil :button :display) ;; check trackpad button to enable/disable arm
    (send self :gripper-cb :larm :wait nil :button :blue) ;; check trigger button to close gripper
    (send self :gripper-cb :rarm :wait nil :button :blue) ;; check trigger button to close gripper
    ;; TODO
    ;; (send self :record-cb :button :display) ;; check menu button to start/stop recording
    ;; (send self :remove-cb :button :display) ;; check menu button to remove rosbag recording
    (send self :init-cb :larm :button :red) ;; check grip button to initialize robot
    (send self :init-cb :rarm :button :red))
  (:rogik-loop-enable-arm (arm &key (mirror nil) (button :display))
    (let ((enable-button-p
            (send self :controller-button-p
                  (if mirror (send self :get-opposite-arm arm) arm) button
                  :button-toggle t))
          (enable-p (send self :get-arm-val arm :enable)))
      (if enable-button-p
        (progn
          (if enable-p
            (progn
              (send robot-interface :cancel-angle-vector
                    :controller-type (send self :get-arm-controller arm))
              (if (find-method robot-interface :torso-controller)
                (send robot-interface :cancel-angle-vector :controller-type :torso-controller))
              (ros::ros-info "arm: ~A stopping" arm))
            (progn
              (send self :update-robot)
              (ros::ros-info "arm: ~A starting" arm)
              (send self :calib-controller arm)))
          (send self :set-arm-val arm :enable (not enable-p))))))
  (:get-base->gripper-coords (arm &key (wait-time 0.1))
    (let ((rogik-frame-id (send self :get-arm-val arm :rogik-frame-id))
          (start-time (ros::time 0))
          coords)
      (if (or (null base-frame-id) (null rogik-frame-id))
        (progn
          (ros::ros-warn "Frame ids are not set properly.")
          (ros::ros-warn "base-frame-id: ~A" base-frame-id)
          (ros::ros-warn "rogik-frame-id: ~A" rogik-frame-id)
          (return-from :get-base->rogik-coords nil)))
      (send tfl :wait-for-transform base-frame-id rogik-frame-id start-time wait-time)
      (setq coords (send tfl :lookup-transform base-frame-id rogik-frame-id start-time))
      coords))
  (:controller-blue-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          (if (eq arm :larm) 'larm-blue 'rarm-blue)
          :button-toggle button-toggle))
  (:controller-red-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          (if (eq arm :larm) 'larm-red 'rarm-red)
          :button-toggle button-toggle))
  (:controller-display-button-p (arm &key (button-toggle t))
    (send self :controller-button-common-p
          (if (eq arm :larm) 'larm-display 'rarm-display)
          :button-toggle button-toggle)))

(provide :robot-teaching-interface)
