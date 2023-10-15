#!/usr/bin/env roseus
(require :robot-teleop-interface "package://eus_imitation/euslisp/robot-teleop-interface.l")

(defclass robot-imitation-interface
  :super robot-teleop-interface
  :slots (;; eus interface
          robot
          robot-interface
          robot-name
          draw-object-list

          ;; ROS settings
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

          ;; robot state and action
          robot-action
          robot-state
          larm-current-end-coords
          larm-current-end-pos
          larm-current-end-rpy
          larm-joint-state
          larm-grasp
          larm-gripper-pos
          rarm-current-end-coords
          rarm-current-end-pos
          rarm-current-end-rpy
          rarm-joint-state
          rarm-grasp
          rarm-gripper-pos
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
          rarm-target-angular-velocity

          ;; IK and pose
          ik-stop-step
          av-tm
          av-scale
          min-time
          init-pose))

(defmethod robot-imitation-interface
  (:init  (&rest args)
   (send-super* :init args)
   (send self :set-val 'robot-interface *ri*) ;; set robot-interface
   (send self :set-val 'robot-action nil)
   (send self :set-val 'robot-state nil)
   (send self :set-val 'recording nil)
   (send self :set-val 'control-arm nil)
   (send self :set-val 'loop-enable nil)

   ;; IK and control
   (send self :set-val 'ik-stop-step 10)
   (send self :set-val 'av-tm 800)
   (send self :set-val 'av-scale 5.0)
   (send self :set-val 'min-time 0.3)
   (send self :set-val 'gripper-threshold 0.5)

   ;; ROS
   (send self :set-val 'node-name "eus_imitation")
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
    (send-super :ros-init)
    (ros::advertise robot-state-topic-name eus_imitation::Float32MultiArrayStamped 1)
    (ros::advertise robot-larm-state-topic-name eus_imitation::Float32MultiArrayStamped 1)
    (ros::advertise robot-rarm-state-topic-name eus_imitation::Float32MultiArrayStamped 1)
    ;; (if (send self :get-val 'data-collection)
    ;;   (ros::advertise robot-action-topic-name eus_imitation::Float32MultiArrayStamped 1))
    (if (send self :get-val 'data-collection) ;; NOTE pub robot action when data collection, sub when imitation
      (ros::advertise robot-action-topic-name eus_imitation::Float32MultiArrayStamped 1)
      (ros::subscribe robot-action-topic-name eus_imitation::Float32MultiArrayStamped #'(lambda (msg) (send self :set-val 'robot-action (send msg :data))) 1))
    t)
  (:reset-arm-val (arm)
    (send-super :reset-arm-val arm)
    (send self :set-arm-val arm :enable nil)
    (send self :set-arm-val arm :grasp 0)
    (send self :set-arm-val arm :current-end-coords nil)
    (send self :set-arm-val arm :gripper-pos 0)
    (send self :set-arm-val arm :target-gripper-pos 0.09)
    (send self :set-arm-val arm :current-end-pos nil)
    (send self :set-arm-val arm :current-end-rpy nil))

  (:dump-pose (pose &key (pose-name "init_pose"))
    (let ((package-path (ros::rospack-find "eus_imitation")))
      (with-open-file (str (concatenate string package-path "/config/" robot-name "/" pose-name ".yaml" )
                      :direction :output
                      :if-exists :supersede
                      :if-does-not-exist :create)
        (format str "~A:~%" pose-name)
        (dotimes (i (length pose))
          (format str " - ~A ~%" (elt pose i))))
      (ros::ros-info "Dump pose : ~A" pose)))
  (:set-init-pose (init-pose)
    (send self :dump-pose init-pose :pose-name "init_pose")
    (ros::set-param "/eus_imitation/init_pose" (coerce init-pose cons))
    (ros::ros-info "Initial Pose setted and dumped : ~A" init-pose))
  (:load-init-pose ()
    (let (init-pose
           (package-path (ros::rospack-find "eus_imitation")))
    (if (setq init-pose (ros::get-param "/eus_imitation/init_pose"))
        (progn
          (setq init-pose (coerce init-pose float-vector))
          (ros::ros-info "Loaded init pose : ~A" init-pose))
        (progn
          (setq init-pose (send robot :reset-pose))
          (send self :set-init-pose init-pose)))
    (send self :set-val 'init-pose init-pose)))
  (:init-pose ()
    (send self :load-init-pose)
    (send self :update-robot)
    (send robot-interface :angle-vector init-pose 3000)
    (send robot-interface :wait-interpolation)
    (if (send robot :larm)
        (send self :stop-grasp :larm :wait t))
    (if (send robot :rarm)
        (send self :stop-grasp :rarm :wait t)))
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
            (setq scaled-current-pos (* current-pos (/ 0.09 81.7714))) ;; TODO this is for PR2
            (setq target-pos (+ scaled-current-pos pos)))
          (setq target-pos pos))
        (if (> target-pos 0.09) (setq target-pos 0.09))
        (if (< target-pos 0.0) (setq target-pos 0.0))
        (send self :set-arm-val arm :target-gripper-pos target-pos)
        (send robot-interface :move-gripper arm target-pos :effort (* 2000 g) :wait wait)))
  (:update-robot-action () ;; when data collection
    (ros::ros-error ":update-robot-action should be defined by user."))
  (:update-robot-state ()
    (ros::ros-error ":update-robot-state should be defined by user."))
  (:parse-robot-action () ;; parse action from ros topic and execute it when imitation mode. define own action execution
    (ros::ros-error ":parse-robot-action should be defined by user."))
  (:move-robot () ;; move real robot
      (if (and data-collection control-arm)
        (progn ;; when data collection is enabled and arm control is enabled
          (if (eq control-arm :arms)
                (progn ;; when both arms are enabled
                  (send self :get-target-coords :larm :pos-scale pos-scale :rot-scale rot-scale)
                  (send self :get-target-coords :rarm :pos-scale pos-scale :rot-scale rot-scale)
                  (send self :inverse-kinematics :arms (list (send self :get-arm-val :larm :target-coords) (send self :get-arm-val :rarm :target-coords))))
                (progn ;; when one arm is enabled
                  (send self :get-target-coords control-arm :pos-scale pos-scale :rot-scale rot-scale)
                  (send self :inverse-kinematics control-arm (send self :get-arm-val control-arm :target-coords))
                  (send self :update-robot-action)
                  (send self :publish-robot-action)))
          (send self :angle-command-vector control-arm :torso nil)
          (if (> (send self :get-arm-val :larm :grasp) (send self :get-val 'gripper-threshold))
              (send self :start-grasp :larm :wait nil)
              (send self :stop-grasp :larm :wait nil))
          (if (> (send self :get-arm-val :rarm :grasp) (send self :get-val 'gripper-threshold))
              (send self :start-grasp :rarm :wait nil)
              (send self :stop-grasp :rarm :wait nil)))
        (if (and (not data-collection) (send self :get-val 'robot-action)) ;; when imitation, parsing action from topic recieved and execute
          (send self :parse-robot-action))))
  (:main-loop-step ()
      (send self :button-cb)
      (send self :update-robot)
      (send self :publish-robot-state)
      (send *irtviewer* :draw-objects)
      (send self :move-robot))
  (:main-loop (&key (imitation nil))
    (send self :update-robot)
    (send self :init-pose)
    ;; (if imitation ;; TODO
    ;;     (progn
    ;;       (ros::ros-info "Imitation control enabled")
    ;;       (ros::subscribe robot-action-topic-name eus_imitation::Float32MultiArrayStamped #'(lambda (msg) (send self :set-val 'robot-action (send msg :data))) 1)))
    (while (ros::ok)
      (ros::rate rate) ;; need to set every loop, otherwise it will be too fast. I don't know why.
      (send self :main-loop-step)
      (ros::spin-once)
      (ros::sleep))))

(provide :robot-imitation-interface)
