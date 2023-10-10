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

(defclass robot-teleop-interface
  :super propertied-object
  :slots (
          ;; eus interface
          robot
          robot-interface
          draw-object-list

          ;; define robot action and state (proprioception)
          robot-action ;; action representated as a float vector
          robot-action-type ;; end-effector pose or joint angle TODO
          robot-state ;; state representated as a float vector
          robot-state-type ;; end-effector pose or joint angle or something else TODO
          ;; joint-states
          joint-pos
          joint-vel
          joint-efforts
          control-arm


          ;; ros settings
          node-name
          robot-action-topic-name
          robot-state-topic-name
          robot-larm-state-topic-name
          robot-rarm-state-topic-name
          robot-joint-states-topic-name
          record-trigger-service-name
          remove-trigger-service-name
          larm-gripper-status-topic-name
          rarm-gripper-status-topic-name
          rate ;; control rate

          ;; robot control states
          control-mode ;; RL, IL control or teleop for collecting demomstration
          larm-enable
          rarm-enable
          base-enable-p ;; TODO
          torse-enable-p ;; TODO
          larm-enable-p
          rarm-enable-p
          larm-grasp
          rarm-grasp
          larm-current-end-coords
          larm-current-end-pos
          larm-current-end-rpy
          rarm-current-end-coords
          rarm-current-end-pos
          rarm-current-end-rpy
          recording
          tracking
          larm-tracking
          rarm-tracking
          loop-enable
          larm-loop-enable
          rarm-loop-enable
          larm-gripper-pos
          rarm-gripper-pos
          control-joint-list
          larm-joint-state
          rarm-joint-state
          data-collection ;; for collecting demonstration

          ;; for IK and pose
          scale
          ik-stop-step
          av-tm
          av-scale
          min-time
          init-pose
          reset-pose ;; initial pose for RL or IL

          larm-target-coords
          rarm-target-coords
          larm-target-end-pos
          rarm-target-end-pos
          larm-target-end-rpy
          rarm-target-end-rpy
          larm-target-gripper-pos
          rarm-target-gripper-pos
          larm-target-linear-velocity
          larm-target-angular-velocity
          rarm-target-linear-velocity
          rarm-target-angular-velocity))

(defmethod robot-teleop-interface
  (:init  (&key (imitation nil) (device-type nil))

   ;; set eus robot interface
   (send self :set-val 'robot-interface *ri*)
   (send self :set-val 'data-collection (not imitation)) ;; mode
   (send self :set-val 'robot-action nil)
   (send self :set-val 'control-arm nil)
   (send self :set-val 'loop-enable nil)

   ;; for IK
   (send self :set-val 'ik-stop-step 10)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)
   (send self :set-val 'scale 1.0)
   ;; (send self :set-val 'av-tm 200)
   ;; (send self :set-val 'av-scale 2.0)
   ;; (send self :set-val 'min-time 0.1)
    ;; (send self :set-val 'av-tm 300)



   ;; prepare for ros init
   (send self :set-val 'node-name "eus_imitation")
   (send self :set-val 'rate 10)
   (send self :set-val 'robot-action-topic-name "/eus_imitation/robot_action")
   (send self :set-val 'robot-state-topic-name "/eus_imitation/robot_state")
   (send self :set-val 'robot-larm-state-topic-name "/eus_imitation/l_arm_state")
   (send self :set-val 'robot-rarm-state-topic-name "/eus_imitation/r_arm_state")
   (send self :set-val 'robot-joint-states-topic-name "/joint_states")
   (send self :set-val 'record-trigger-service-name "/eus_imitation/rosbag_record_trigger")
   (send self :set-val 'remove-trigger-service-name "/eus_imitation/rosbag_remove_trigger")
   (send self :reset)

    self)
  (:ros-init ()
    (ros::advertise robot-state-topic-name eus_imitation::Float32MultiArrayStamped 1)
    (ros::advertise robot-larm-state-topic-name eus_imitation::Float32MultiArrayStamped 1)
    (ros::advertise robot-rarm-state-topic-name eus_imitation::Float32MultiArrayStamped 1)

    ;; publish action topic when data collection
    ;; subscribe action topic when imitation
    (if (send self :get-val 'data-collection)
      (ros::advertise robot-action-topic-name eus_imitation::Float32MultiArrayStamped 1))
    ;; (ros::subscribe robot-joint-states-topic-name
    ;;                 sensor_msgs::JointState #'send self :joint-states-cb) ;; TODO currently only rarm
    t)

  ;; (:joint-states-cb (msg)
  ;;   (send self :set-val 'joint-pos (send msg :position))
  ;;   (send self :set-val 'joint-vel (send msg :velocity))
  ;;   (send self :set-val 'joint-efforts (send msg :effort)))

  (:reset ()
    (send self :reset-arm-val :larm)
    (send self :reset-arm-val :rarm))
  (:reset-arm-val (arm)
    (send self :set-arm-val arm :enable nil)
    (send self :set-arm-val arm :grasp 0)
    (send self :set-arm-val arm :current-end-coords nil)
    (send self :set-arm-val arm :gripper-pos 0)
    (send self :set-arm-val arm :target-gripper-pos 0.09)

    (send self :set-val 'tracking nil)
    (send self :set-arm-val :larm :tracking nil)
    (send self :set-arm-val :rarm :tracking nil)

    (send self :set-arm-val arm :current-end-pos nil)
    (send self :set-arm-val arm :current-end-rpy nil))

  (:set-init-pose ()
    (send self :update-robot)
    (setq *init-pose* (send robot :angle-vector))
    (send self :set-val 'init-pose *init-pose*)
    (dump-loadable-structure "init-pose.l" *init-pose*)
    (ros::ros-info "Set init pose : ~A" *init-pose*))
  (:load-init-pose ()
    (load "init-pose.l") ;; TODO from yaml
    (ros::ros-info "Loaded init pose is ~A" *init-pose*)
    (send self :set-val 'init-pose *init-pose*))
  (:init-pose ()
    (send self :load-init-pose)
    (send self :update-robot)
    (send robot-interface :angle-vector init-pose 3000)
    (if (send robot :larm)
        (send self :stop-grasp :larm :wait t))
    (if (send robot :rarm)
        (send self :stop-grasp :rarm :wait t))
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

  (:update-robot-model ()
    (send robot :angle-vector (send robot-interface :state :potentio-vector)))
  (:update-current-end-coords ()
        (if (send robot :larm)
            (progn
                (send self :set-arm-val :larm :current-end-coords
                        (send robot :larm :end-coords :copy-worldcoords))
                (send self :set-arm-val :larm :current-end-pos
                        (send (send self :get-arm-val :larm :current-end-coords) :worldpos))
                (send self :set-arm-val :larm :current-end-rpy
                        (coerce (elt (rpy-angle (send (send self :get-arm-val :larm :current-end-coords) :worldrot)) 0) float-vector))))
        (if (send robot :rarm)
            (progn
                (send self :set-arm-val :rarm :current-end-coords
                        (send robot :rarm :end-coords :copy-worldcoords))
                (send self :set-arm-val :rarm :current-end-pos
                        (send (send self :get-arm-val :rarm :current-end-coords) :worldpos))
                (send self :set-arm-val :rarm :current-end-rpy
                        (coerce (elt (rpy-angle (send (send self :get-arm-val :rarm :current-end-coords) :worldrot)) 0) float-vector)))))
  (:update-current-gripper-pos ()
        (if (send robot :larm)
            (send self :set-arm-val :larm :gripper-pos
                    (send robot-interface :gripper :larm :position)))
        (if (send robot :rarm)
            (send self :set-arm-val :rarm :gripper-pos
                    (send robot-interface :gripper :rarm :position))))
  (:update-current-joint-state ()
        (if (send robot :larm)
            (send self :set-arm-val :larm :joint-state
                    (send robot :larm :angle-vector)))
        (if (send robot :rarm)
            (send self :set-arm-val :rarm :joint-state
                    (send robot :rarm :angle-vector))))
  (:update-robot ()
    (send self :update-robot-model)
    (send self :update-current-end-coords)
    (send self :update-current-gripper-pos)
    (send self :update-current-joint-state)
    (send self :update-robot-state)) ;; update robot state for IL or RL, defined by user
  (:update-robot-state () ;; this should be defined by user, as an abstract method. here is an example.
      (let (end-effector-pos joint-state gripper-pos)
        (setq end-effector-pos (send (send (send self :get-arm-val :rarm :current-end-coords) :copy-worldcoords) :worldpos))
        (setq gripper-pos (float-vector (send self :get-arm-val :rarm :gripper-pos)))
        (send self :set-val 'robot-state (concatenate float-vector end-effector-pos gripper-pos))))
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
     (send robot-interface :start-grasp arm :wait wait))
  (:stop-grasp (arm &key (wait nil))
     (send robot-interface :stop-grasp arm :wait wait))
  (:move-gripper (arm pos &key (delta nil) (g 0.01) (wait nil) ) ;; open : pos-abs=0.0, close : pos-abs=0.09
      (let (current-pos scaled-current-pos target-pos )
        (if delta
          (progn
            (setq current-pos (send self :get-arm-val arm :gripper-pos))
            (setq scaled-current-pos (* current-pos (/ 0.09 81.7714))) ;; scale pos 81.7714 -> 0.09, 0.0 -> 0.0
            (setq target-pos (+ scaled-current-pos pos)))
          (setq target-pos pos))
        (if (> target-pos 0.09) (setq target-pos 0.09))
        (if (< target-pos 0.0) (setq target-pos 0.0))
        (send self :set-arm-val arm :target-gripper-pos target-pos)
        (send robot-interface :move-gripper arm target-pos :effort (* 2000 g) :wait wait)))
  (:calib-controller ()
    (ros::ros-error ":calib-controller is not implemented."))
  (:get-target-coords (arm &key (pos-scale 1.5))
    (ros::ros-error ":get-target-coords is not implemented."))
  (:controller-cb ()
    (ros::ros-error ":controller-cb is not implemented."))
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
    (send self :loop-enable-arm arm :mirror mirror :button button)
    (if (and (send self :get-arm-val :larm :enable) (send self :get-arm-val :rarm :enable))
      (send self :set-val 'control-arm :arms)
      (if (send self :get-arm-val :larm :enable)
        (send self :set-val 'control-arm :larm)
        (if (send self :get-arm-val :rarm :enable)
          (send self :set-val 'control-arm :rarm)
          (send self :set-val 'control-arm nil)))))
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
              (ros::ros-info "arm: ~A stopping" arm))
            (progn
              (send self :update-robot)
              (ros::ros-info "arm: ~A starting" arm)
              (send self :calib-controller arm)))
          (send self :set-arm-val arm :enable (not enable-p))))))
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
            (print "test")
            (print arm)
            (print button)
            (send self :init-pose)
            (send self :reset)
            (send self :update-robot)
            (ros::ros-info "Initiailized")))))
  (:gripper-cb (arm &key (wait nil) (button :trigger))
    (if (send self :get-arm-val arm button) ;; NOTE controller-button-p is not used cause it is not toggle
        (send self :set-arm-val arm :grasp 1) ;;  we can set this t or nil but in case of use of this variable as state
        (send self :set-arm-val arm :grasp 0)))
  (:button-cb ()
    (ros::ros-error ":button-cb is not implemented."))
  (:publish-robot-state ()
    (let ((robot-state-msg (instance eus_imitation::Float32MultiArrayStamped :init))
          (robot-larm-state-msg (instance eus_imitation::Float32MultiArrayStamped :init))
          (robot-rarm-state-msg (instance eus_imitation::Float32MultiArrayStamped :init)))
      (send robot-state-msg :data (send self :get-val 'robot-state))
      (send robot-state-msg :header :stamp (ros::time-now))
      (ros::publish robot-state-topic-name robot-state-msg)
      (if (send robot :larm)
          (progn
              (send robot-larm-state-msg :data (concatenate float-vector (send self :get-arm-val :larm :joint-state) (float-vector (send self :get-arm-val :larm :gripper-pos))))
              (send robot-larm-state-msg :header :stamp (ros::time-now))
              (ros::publish robot-larm-state-topic-name robot-larm-state-msg)))
      (if (send robot :rarm)
          (progn
              (send robot-rarm-state-msg :data (concatenate float-vector (send self :get-arm-val :rarm :joint-state) (float-vector (send self :get-arm-val :rarm :gripper-pos))))
              (send robot-rarm-state-msg :header :stamp (ros::time-now))
              (ros::publish robot-rarm-state-topic-name robot-rarm-state-msg)))))
  (:publish-robot-action ()
    (let ((robot-action-msg (instance eus_imitation::Float32MultiArrayStamped :init)))
      (send robot-action-msg :data (send self :get-val 'robot-action))
      (send robot-action-msg :header :stamp (ros::time-now))
      (ros::publish robot-action-topic-name robot-action-msg)))
  (:main-loop-step ()
      (send self :button-cb)
      (send self :update-robot)
      (send self :publish-robot-state)
      (send *irtviewer* :draw-objects)
      (send self :move-robot))
  (:main-loop (&key (imitation nil))
    (send self :update-robot)
    (send self :init-pose)
    (if imitation ;; TODO
        (progn
          (ros::ros-info "Imitation control enabled")
          (ros::subscribe robot-action-topic-name eus_imitation::Float32MultiArrayStamped #'(lambda (msg) (send self :set-val 'robot-action (send msg :data))) 1)))
    (while (ros::ok)
      (ros::rate rate) ;; need to set every loop, otherwise it will be too fast. I don't know why.
      (send self :main-loop-step)
      (ros::spin-once)
      (ros::sleep))))

(provide :robot-teleop-interface)