#!/usr/bin/env roseus
(require :pr2-interface "package://pr2eus/pr2-interface.l")
;; (require :display-robot-state "package://jsk_interactive_marker/euslisp/display-robot-state.l")
;;

;; (load "package://informatized_pr2_data_augmentation/euslisp/pr2-utils.l")
;; (load "package://informatized_body_core/euslisp/informatized-body-core.l")
(ros::roseus-add-msgs "informatized_body_msgs")
(ros::roseus-add-msgs "eus_imitation")

(ros::load-ros-manifest "roseus")
(ros::roseus-add-msgs "sensor_msgs")
(ros::roseus "eus_imitation" :anonymous t)
;; (ros::advertise "/eus_imitation/robot_action" eus_imitation::Float32MultiArrayStamped 1)
;; (ros::advertise "/eus_imitation/robot_state" eus_imitation::Float32MultiArrayStamped 1)
(ros::advertise "/eus_imitation/target_coords" eus_imitation::Float32MultiArrayStamped 1)
(ros::advertise "/eus_imitation/controller/target_state" informatized_body_msgs::TargetState 1)

(ros::set-logger-level "ros.roseus" 5)
(ros::load-ros-manifest "actionlib_msgs")
;; (ros::load-ros-manifest "eus_imitation")
(ros::load-ros-manifest "moveit_msgs")
(ros::load-ros-manifest "sensor_msgs")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "std_srvs")


(defclass pr2-imitation-interface
  :super propertied-object
  :slots (robot
          robot-interface
          draw-object-list

          control-mode ;; RL, IL control or teleop for collecting demomstration
          robot-action ;; for RL or IL
          robot-action-type ;; EndEffectorPose (x,y,z,roll,pitch,yaw) or JointAngles (deg or rad) + Gripper (open or close)
          ;; robot-state-topic-name ;; for RL or IL state, can be list of multiple topics
          ;; ;; robot-obs-topic-name ;; for RL or IL observation, can be list of multiple topics

          reset-pose ;; initial pose for RL or IL
          loop-enable

          ;; frame
          tfl
          base-frame-id
          head-frame-id


          ;; for vive control
          ;; larm-tf-world->controllerref
          ;; rarm-tf-world->controllerref
          larm-tf-vrbase->controllerref
          larm-tf-base->gripperref
          rarm-tf-vrbase->controllerref
          rarm-tf-base->gripperref
          tracking
          larm-tracking
          rarm-tracking

          init-pose




          robot-action-topic-name
          robot-state-topic-name
          pr2-imitation-topic-name
          node-name
          recording

          record-trigger-service-name
          remove-trigger-service-name

          larm-gripper-status-topic-name
          rarm-gripper-status-topic-name

          rate ;; control rate

          larm-enable
          rarm-enable
          base-enable-p ;; TODO
          torse-enable-p ;; TODO
          larm-enable-p
          rarm-enable-p
          larm-grasp
          rarm-grasp
          larm-grasping-p
          rarm-grasping-p
          larm-current-end-coords
          rarm-current-end-coords

          larm-loop-enable
          rarm-loop-enable

          control-joint-list



          scale

          arm-cb-solve-ik
          ik-stop-step
          av-tm
          av-scale
          min-time

          ;; for vive interface
          vrbase-device-id
          larm-controller-device-id
          rarm-controller-device-id
          vrbase-frame-id
          larm-controller-frame-id
          rarm-controller-frame-id
          feedback-topic-name
          trigger-wait-count
          menu-wait-count
          trackpad-wait-count
          grip-wait-count
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

          ))



(defmethod pr2-imitation-interface
  (:init  (&key (loop-enable t))
   ;; set
   (send self :set-val 'robot *pr2*)
   (send self :set-val 'robot-interface *ri*)



   ;; for frame
   (send self :set-val 'base-frame-id "base_footprint")
   (send self :set-val 'head-frame-id "head_tilt_link")

   ;; for ik
   (send self :set-val 'ik-stop-step 10)
   ;; (send self :set-val 'av-tm 200)
   ;; (send self :set-val 'av-scale 2.0)
   ;; (send self :set-val 'min-time 0.1)
   (send self :set-val 'scale 1.5)


    ;; (send self :set-val 'av-tm 300)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)

   ;; for workspace
   (send self :set-val 'rate 10)
    (send self :set-val 'tfl (instance ros::transform-listener :init))

    (send self :set-val 'vrbase-frame-id (format nil "lighthouse_LHB_~A" "8E924CC3"))
    ;; (send self :set-arm-val :larm :controller-device-id "FD35BD42")
    ;; (send self :set-arm-val :rarm :controller-device-id "F7AFBF47")

    (send self :set-arm-val :larm :controller-device-id "F7AFBF47")
    (send self :set-arm-val :rarm :controller-device-id "FD35BD42")


    (send self :set-val 'tracking nil)
    (send self :set-val 'recording nil)
    (send self :set-arm-val :larm :tracking nil)
    (send self :set-arm-val :rarm :tracking nil)


    (send self :set-val 'trigger-wait-count 5)
    (send self :set-val 'menu-wait-count 5)
    (send self :set-val 'trackpad-wait-count 5)
    (send self :set-val 'grip-wait-count 5)


    (send self :set-arm-val :larm :gripper-status-topic-name
            "/l_gripper_controller/gripper_action/status")
    (send self :set-arm-val :rarm :gripper-status-topic-name
            "/r_gripper_controller/gripper_action/status")

   (send self :set-val 'record-trigger-service-name "/rosbag_record_trigger")
   (send self :set-val 'remove-trigger-service-name "/rosbag_remove_trigger")
   (send self :set-val 'robot-action-topic-name "/eus_imitation/robot_action")
   (send self :set-val 'robot-state-topic-name "/eus_imitation/robot_state")
   (send self :set-val 'node-name "eus_imitation")

   (send self :reset)


    self)
  (:ros-init ()


    (ros::advertise robot-action-topic-name eus_imitation::Float32MultiArrayStamped 1)
    (ros::advertise robot-state-topic-name eus_imitation::Float32MultiArrayStamped 1)
    (if larm-controller-device-id
      (progn
        (send self :set-arm-val :larm :controller-frame-id
              (format nil "controller_LHR_~A" larm-controller-device-id))
        (ros::subscribe (format nil "/~A/joy"
                                (send self :get-arm-val :larm :controller-frame-id))
                        sensor_msgs::Joy #'send self :controller-button-cb :larm)
        (ros::ros-info "Detected left arm controller device ID :  ~A" larm-controller-device-id))
      (ros::ros-warn "param: /eus_vive/l_controller_id is not set"))
    (if rarm-controller-device-id
      (progn
        (send self :set-arm-val :rarm :controller-frame-id
              (format nil "controller_LHR_~A" rarm-controller-device-id))
        (ros::subscribe (format nil "/~A/joy"
                                (send self :get-arm-val :rarm :controller-frame-id))
                        sensor_msgs::Joy #'send self :controller-button-cb :rarm)
        (ros::ros-info "Detected right arm controller device ID :  ~A" rarm-controller-device-id))
      (ros::ros-warn "param: /eus_vive/r_controller_id is not set"))
    t)
  (:reset ()
    (send self :reset-arm-val :larm)
    (send self :reset-arm-val :rarm))
  (:reset-arm-val (arm)
    (send self :set-arm-val arm :enable nil)
    (send self :set-arm-val arm :grasp nil)
    (send self :set-arm-val arm :grasping-p nil)
    (send self :set-arm-val arm :current-end-coords nil)

    ;; for vive
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
  (:set-init-pose ()
    (setq *init-pose* (send self :update-robot-model))
    (ros::ros-info "Set init pose : ~A" *init-pose*)
    (send self :set-val 'init-pose *init-pose*)
    (dump-loadable-structure "init-pose.l" *init-pose*))
  (:load-init-pose ()
    (load "init-pose.l") ;; TODO from yaml
    (ros::ros-info "Loaded init pose is ~A" *init-pose*)
    (send self :set-val 'init-pose *init-pose*))
  (:init-pose ()
    (send self :load-init-pose)
    (send self :update-robot-model)
    (send robot-interface :angle-vector init-pose 5000)
    (send self :stop-grasp :larm :wait t)
    (send self :stop-grasp :rarm :wait t)
    (send robot-interface :wait-interpolation))

  (:get-arm-controller (arm)
    (cond
      ((eq arm :larm) :larm-controller)
      ((eq arm :rarm) :rarm-controller)
      (t nil)))
  (:get-opposite-arm (arm)
    (cond
      ((eq arm :larm) :rarm)
      ((eq arm :rarm) :larm)
      (t nil)))
  (:get-arm-val (arm name)
    (if (not (or (eq arm :larm) (eq arm :rarm)))
      (progn
        (ros::ros-error (format nil "Invalid args: (send self :get-arm-val ~A ~A)" arm name))
        (return-from :get-arm-val nil)))
    (let ((valname (format nil "~A-~A" (string-upcase arm) (string-upcase name))))
      (if (find (read-from-string valname) (mapcar #'car (cdr (send self :slots))))
        (send self :get-val (intern valname)))
      ))
  (:set-arm-val (arm name val)
    (if (not (or (eq arm :larm) (eq arm :rarm)))
      (progn
        (ros::ros-error (format nil "Invalid args: (send self :set-arm-val: ~A ~A ~A)" arm name val))
        (return-from :set-arm-val nil)))
    (let ((valname (format nil "~A-~A" (string-upcase arm) (string-upcase name))))
      (send self :set-val (intern valname) val)
      ))
  (:update-robot-model ()
    (send robot :angle-vector (send robot-interface :state :potentio-vector)))
  (:update-current-end-coords ()
     (let ((av (send robot :angle-vector)))
       (send robot :angle-vector
             (send robot-interface :state :potentio-vector))
       (send self :set-arm-val :larm :current-end-coords
             (send robot :larm :end-coords :copy-worldcoords))
       (send self :set-arm-val :rarm :current-end-coords
             (send robot :rarm :end-coords :copy-worldcoords))))
  (:inverse-kinematics (arm target-coords &rest args)
    (if (eq arm :arms)
      (send* robot :inverse-kinematics target-coords :rotation-axis (list t t)
             :move-target (list (send robot :larm :end-coords) (send robot :rarm :end-coords))
             :avoid-collision-distance 5 :revert-if-fail nil :stop ik-stop-step
             :debug-view nil args)
      (send* robot :inverse-kinematics target-coords
             :rotation-axis t :move-target (send robot arm :end-coords)
             :avoid-collision-distance 5 :revert-if-fail nil :stop ik-stop-step
             :debug-view nil args)))
  (:angle-vector (&rest args)
    (send* robot-interface :angle-vector args))
  (:angle-command-vector (arm &key (torso nil))
    (cond
      ((equal arm :arms)
       (send self :angle-vector (send robot :angle-vector)
             av-tm (send self :get-arm-controller :larm) 0
             :min-time min-time :scale av-scale)
       (send self :angle-vector (send robot :angle-vector)
             av-tm (send self :get-arm-controller :rarm) 0
             :min-time min-time :scale av-scale)
       (if torso
         (send self :angle-vector (send robot :angle-vector)
               torso-av-tm :torso-controller 0
               :min-time torso-min-time :scale torso-av-scale)))
      ((equal arm :larm)
       (send self :angle-vector (send robot :angle-vector)
             av-tm (send self :get-arm-controller :larm) 0
             :min-time min-time :scale av-scale)
       (if torso
         (send self :angle-vector (send robot :angle-vector)
               torso-av-tm :torso-controller 0
               :min-time torso-min-time :scale torso-av-scale)))
      ((equal arm :rarm)
       (send self :angle-vector (send robot :angle-vector)
             av-tm (send self :get-arm-controller :rarm) 0
             :min-time min-time :scale av-scale)
       (if torso
         (send self :angle-vector (send robot :angle-vector)
               torso-av-tm :torso-controller 0
               :min-time torso-min-time :scale torso-av-scale)))
      (t nil)))
  (:start-grasp (arm &key (wait nil))
     (send robot-interface :start-grasp arm :wait wait)
     (send self :set-arm-val arm :grasp 1))
  (:stop-grasp (arm &key (wait nil))
     (send robot-interface :stop-grasp arm :wait wait)
     (send self :set-arm-val arm :grasp 0))
  (:calib-controller (arm)
    (let (tf-vrbase->controllerref tf-base->gripperref)
      (ros::ros-info "Calibrating controller ~A" arm)
      (send self :update-current-end-coords)
      (send self :set-arm-val arm :tf-vrbase->controllerref (make-coords
                                         :pos (send (send self :get-vrbase->controller-coords arm) :pos)
                                         :rot (send (send self :get-vrbase->controller-coords arm) :rot)))
      (send self :set-arm-val arm :tf-base->gripperref (make-coords
                                          :pos (send (send self :get-arm-val arm :current-end-coords) :pos)
                                          :rot (send (send self :get-arm-val arm :current-end-coords) :rot)))))
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
      (send tfl :wait-for-transform vrbase-frame-id controller-frame-id start-time wait-time) ;; NOTE this executed
      (setq coords (send tfl :lookup-transform vrbase-frame-id controller-frame-id start-time)) ;; NOTE this executed too
      coords))
  ;; (:get-world->controller-coords (arm &key (wait-trigger nil) (wait-time 0.1))
  ;;   (let ((controller-frame-id (send self :get-arm-val arm :controller-frame-id))
  ;;         (start-time (ros::time 0))
  ;;         coords controller-trigger-button-p)
  ;;     (if (or (null vrbase-frame-id) (null controller-frame-id))
  ;;       (progn
  ;;         (ros::ros-warn "Frame ids are not set properly.")
  ;;         (ros::ros-warn "vrbase-frame-id: ~A" vrbase-frame-id)
  ;;         (ros::ros-warn "controller-frame-id: ~A" controller-frame-id)
  ;;         (return-from :get-vrbase->controller-coords nil)))
  ;;     (if wait-trigger
  ;;       (progn
  ;;         (ros::ros-info "Waiting for trigger: ~A~%" (string-downcase arm))
  ;;         (while (and (ros::ok) (not controller-trigger-button-p))
  ;;           (setq controller-trigger-button-p
  ;;                 (send self :controller-trigger-button-p arm :button-toggle t))
  ;;           (ros::spin-once)
  ;;           (ros::sleep))))
  ;;     (send tfl :wait-for-transform "world" controller-frame-id start-time wait-time) ;; NOTE this executed
  ;;     (setq coords (send tfl :lookup-transform "world" controller-frame-id start-time)) ;; NOTE this executed too
  ;;     coords))
  (:get-target-coords (arm &key (pos-scale 1.5))
    (let* ((tf-vrbase->controller (send self :get-vrbase->controller-coords arm))
          (tf-base->gripperref (send self :get-arm-val arm :tf-base->gripperref))
          (tf-controllerref->controller (send (send (send self :get-arm-val arm :tf-vrbase->controllerref) :transformation tf-vrbase->controller) :copy-worldcoords))
          (tf-gripperref->gripper (make-coords
                                      :pos (scale pos-scale (send tf-controllerref->controller :pos))
                                      :rot (send tf-controllerref->controller :rot)))
          (target-coords (send tf-gripperref->gripper :transform (send tf-base->gripperref :copy-worldcoords) :world)))
      target-coords))

  ;; (:enable (&key (torso t))
  ;;   (send self :set-val 'loop-enable t)
  ;;   (send self :set-arm-val :larm :loop-enable t)
  ;;   (send self :set-arm-val :rarm :loop-enable t)
  ;;   (send robot-interface :cancel-angle-vector
  ;;         :controller-type (send self :get-arm-controller :larm))
  ;;   (send robot-interface :cancel-angle-vector
  ;;         :controller-type (send self :get-arm-controller :rarm))
  ;;   (if (and torso (find-method robot-interface :torso-controller))
  ;;     (send robot-interface :cancel-angle-vector :controller-type :torso-controller))
  ;;   )
  ;; (:enable-arm (arm)
  ;;   (send robot-interface :cancel-angle-vector
  ;;         :controller-type (send self :get-arm-controller arm))
  ;;   (send self :set-arm-val arm :enable t)
  ;;   )
  ;; (:enable-torso ()
  ;;   (if (find-method robot-interface :torso-controller)
  ;;     (send robot-interface :cancel-angle-vector :controller-type :torso-controller)))
  ;; (:disable (&key (torso t))
  ;;   (send self :set-val 'loop-enable nil)
  ;;   (send self :set-arm-val :larm :loop-enable nil)
  ;;   (send self :set-arm-val :rarm :loop-enable nil)
  ;;   (send robot-interface :cancel-angle-vector
  ;;         :controller-type (send self :get-arm-controller :larm))
  ;;   (send robot-interface :cancel-angle-vector
  ;;         :controller-type (send self :get-arm-controller :rarm))
  ;;   (if (and torso (find-method robot-interface :torso-controller))
  ;;     (send robot-interface :cancel-angle-vector :controller-type :torso-controller))
  ;;   )
  ;; (:disable-arm (arm)
  ;;   (send robot-interface :cancel-angle-vector
  ;;         :controller-type (send self :get-arm-controller arm))
  ;;   (send self :set-arm-val arm :enable nil))
  ;; (:disable-torso ()
  ;;   (if (find-method robot-interface :torso-controller)
  ;;     (send robot-interface :cancel-angle-vector :controller-type :torso-controller)))

  (:controller-button-cb (arm msg)
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
  (:enable-cb (arm &key (mirror nil) (button :trackpad))
    (send self :loop-enable-arm arm :mirror mirror :button button))
  (:loop-enable-arm (arm &key (mirror nil) (button :trackpad))
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
              (ros::ros-info "arm: ~A stopping" arm)
              ;; (send *ri* :speak "Stop Rosbag Record" :wait nil :volume 0.5)
              ;; (data-collection-stop)
              ;; (unix:usleep (* 1000 100))
              (ros::wait-for-service record-trigger-service-name 1)
              (setq req (instance std_srvs::TriggerRequest :init))
              (setq res (ros::service-call record-trigger-service-name req))
              (send self :set-val 'recording nil))
            (progn
              (send self :update-robot-model)
              (ros::ros-info "arm: ~A starting" arm)
              (send self :calib-controller arm)
              ;; (send *ri* :speak "Start Rosbag Record" :wait nil :volume 0.5)
              (ros::wait-for-service record-trigger-service-name 1)
              (setq req (instance std_srvs::TriggerRequest :init))
              (setq res (ros::service-call record-trigger-service-name req nil))
              ;; (unix:usleep (* 1000 1000))
              ;; (data-collection-start :filename nil)
              (send self :set-val 'recording t)))
          (send self :set-arm-val arm :enable (not enable-p))))))
  (:record-cb (&key (button :menu))
    (let (req res (record-button-p
            (send self :controller-button-p :rarm button :button-toggle t))
          (record-p (send self :get-val 'recording)))
      (if record-button-p
          (progn
            (if record-p
              (progn
                (ros::wait-for-service record-trigger-service-name 1)
                (setq req (instance std_srvs::TriggerRequest :init))
                (setq res (ros::service-call record-trigger-service-name req))
                (send self :set-val 'recording nil))
              (progn
                (ros::wait-for-service record-trigger-service-name 1)
                (setq req (instance std_srvs::TriggerRequest :init))
                (setq res (ros::service-call record-trigger-service-name req nil))
                (send self :set-val 'recording t)))
            (send self :set-val 'recording (not record-p))))))
  (:init-cb (&key (button :grip))
    (let ((init-button-p
            (send self :controller-button-p :rarm button :button-toggle t)))
      (if init-button-p
          (progn
            (ros::ros-info "Inititalizing robot...")
            (send self :init-pose)
            (send self :reset)
            (send self :update-robot-model)
            (ros::ros-info "Initiailized")))))
  (:gripper-cb (arm &key (wait nil) (button :trigger))
    (if (and (send self :get-arm-val arm :enable) (send self :get-arm-val arm button))
        (send self :start-grasp arm :wait wait)
        (send self :stop-grasp arm :wait wait)))
  (:remove-cb (&key (button :menu))
    (let (req res (remove-button-p
            (send self :controller-button-p :larm button :button-toggle t)))
      (if remove-button-p
          (progn
            (ros::wait-for-service remove-trigger-service-name 1)
            (setq req (instance std_srvs::TriggerRequest :init))
            (setq res (ros::service-call remove-trigger-service-name req))))))
  (:button-cb (arm)
    ;; (send self :enable-cb :rarm :mirror nil :button :trackpad)
    (send self :enable-cb arm :mirror nil :button :trackpad)
    (send self :record-cb :button :menu)
    (send self :init-cb :button :grip)
    (send self :gripper-cb arm :wait nil :button :trigger)
    (send self :remove-cb :button :menu))
  (:robot-state-publisher (robot-state rot)
    (if rot
        (progn
          (setq rpy-rots (elt (rpy-angle (send robot-state :worldrot)) 0))
          (setq robot-state-data (concatenate float-vector (send robot-state :worldpos) rpy-rots (float-vector (send self :get-arm-val :rarm :grasp)))))
        (setq robot-state-data (concatenate float-vector (send robot-state :worldpos) (float-vector (send self :get-arm-val :rarm :grasp)))))
    (setq robot-state-msg (instance eus_imitation::Float32MultiArrayStamped :init))
    (send robot-state-msg :data robot-state-data)
    (send robot-state-msg :header :stamp (ros::time-now))
    (ros::publish robot-state-topic-name robot-state-msg))
  (:robot-action-publisher (robot-action rot)
    (if rot
        (progn
          (setq rpy-rots (elt (rpy-angle (send robot-action :worldrot)) 0))
          (setq robot-action-data (concatenate float-vector (send robot-action :worldpos) rpy-rots (float-vector (send self :get-arm-val :rarm :grasp)))))
        (setq robot-action-data (concatenate float-vector (send robot-action :worldpos) (float-vector (send self :get-arm-val :rarm :grasp)))))
    (setq robot-action-msg (instance eus_imitation::Float32MultiArrayStamped :init))
    (send robot-action-msg :data robot-action-data)
    (send robot-action-msg :header :stamp (ros::time-now))
    (ros::publish robot-action-topic-name robot-action-msg))
  (:main-loop-step (&key (arm :rarm) (imitation nil) (rot nil))
      (let ((ref-coords) (target-pos) (target-rpy) (robot-action) (robot-state))
      ;; (unless imitation
      ;;   (send self :button-cb arm))

      (if imitation
          (progn
          (setq gripper-action (if (> (elt robot-action-real 3) 0.7) 1 0))
          (if (> gripper-action 0.5)
              (send self :start-grasp arm :wait nil)
              (send self :stop-grasp arm :wait nil)))
          (send self :button-cb arm))


      ;; (send self :button-cb arm)
      (if (send self :get-arm-val arm :enable)
        (progn
          (send self :update-current-end-coords)
          (setf robot-state (send (send self :get-arm-val arm :current-end-coords) :copy-worldcoords)) ;; 3 + 1

          (unless rot
            (setf target-rpy (float-vector pi/2 pi/2 -pi/2))
            )

          (if imitation
            (if rot
              (setf ref-coords (make-coords :pos (subseq robot-action-real 0 3) :rpy (subseq robot-action-real 3 6)))
              (setf ref-coords (make-coords :pos (subseq robot-action-real 0 3) :rpy target-rpy)))
            (setf ref-coords (send (send self :get-target-coords arm :pos-scale 1.0) :copy-worldcoords)))

          (setf target-pos (send ref-coords :worldpos))
          (setf target-rpy (elt (rpy-angle (send ref-coords :worldrot)) 0))
          (unless rot
            (setf target-rpy (float-vector pi/2 pi/2 -pi/2))
            )
          (setf target-coords (make-coords :pos target-pos :rpy target-rpy))
          (send self :inverse-kinematics arm target-coords)
          (setf robot-action target-coords) ;; or joint angles
          (send self :robot-state-publisher robot-state rot) ;; state publisher
          (unless imitation
            (send self :robot-action-publisher robot-action rot))
          (send self :angle-command-vector arm :torso nil)))))
  (:main-loop (&key (arm :rarm) (imitation nil) (rot nil))
    (ros::rate rate)
    (send self :update-current-end-coords)
    ;; (if imitation
    ;;     (progn
    ;;       (send self :set-arm-val arm :enable t)
    ;;       (send self :init-pose)
    ;;       (setf robot-action-real (concatenate float-vector (send (send (send self :get-arm-val arm :current-end-coords) :copy-worldcoords) :worldpos ) (float-vector (send self :get-arm-val arm :grasp)))) ;; 3 + 1
    ;;       (ros::subscribe "/eus_imitation/robot_action" eus_imitation::Float32MultiArrayStamped #'(lambda (msg) (setq robot-action-real (send msg :data))) 1)))

    ;; (send *ri* :speak "I'm ready to start." :wait nil :volume 0.6)

    (while (ros::ok)
      (send self :main-loop-step :arm arm :imitation imitation :rot rot)
      (ros::spin-once)
      (ros::sleep)))
)

(defun init-pr2-imitator ()
  (pr2-init)
  (objects (list *pr2*))
  (setq *pr2-imitator* (instance pr2-imitation-interface :init))
  (send *pr2-imitator* :ros-init))
(defun main (&key (arm :rarm) (imitation nil) (rot nil))
  (init-pr2-imitator)
  (send *pr2-imitator* :main-loop :arm arm :imitation imitation :rot rot))
(defun imitation (&key (arm :rarm))
  (init-pr2-imitator)
  (send *pr2-imitator* :imitation-loop :arm arm))