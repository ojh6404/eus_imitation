#!/usr/bin/env roseus
(require :pr2-interface "package://pr2eus/pr2-interface.l")
(ros::roseus-add-msgs "eus_imitation_msgs")
(ros::roseus-add-msgs "sensor_msgs")
(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_srvs")

(setq *enable* nil)

;; example of euslisp interface for imitation learning using pr2-interface
;; if you want to collect data for imitation learning, set imitation-mode to nil.
;; if imitation-mode is t, it will subscribe robot-action-topic-name and set robot-action to the received message.

;; state is x, y, z, roll, pitch, yaw, gripper position of right arm
;; action is x, y, z, roll, pitch, yaw (in delta), gripper command (0: open, 1: close) of right arm

(defun inverse-kinematics (target-coords &rest args)
    (send* *pr2* :inverse-kinematics target-coords
            :rotation-axis t :move-target (send *pr2* :rarm :end-coords)
            :avoid-collision-distance 5 :revert-if-fail nil :stop 5
            :debug-view nil args))

(defun update-robot ()
    ;; get potentio vector and update robot interface
    (send *pr2* :angle-vector (send *ri* :state :potentio-vector))

    ;; update robot state (in this case, x, y, z, roll, pitch, yaw, gripper position)
    (setq *rarm-end-coords* (send *pr2* :rarm :end-coords :copy-worldcoords))
    (setq *rarm-end-pos* (send *rarm-end-coords* :worldpos))
    (setq *rarm-end-rpy* (coerce (elt
                           (rpy-angle (send *rarm-end-coords* :worldrot)) 0) float-vector))
    (setq *rarm-gripper-pos* (send *ri* :gripper :rarm :position))
    (setq *robot-state* (concatenate float-vector
                                     *rarm-end-pos*
                                     *rarm-end-rpy*
                                     (float-vector *rarm-gripper-pos*)))
    (send *irtviewer* :draw-objects)    ;; draw robot with current state

    ;; publish robot state
    (setq *robot-state-msg* (instance eus_imitation_msgs::FloatVector :init))
    (send *robot-state-msg* :data *robot-state*)
    (send *robot-state-msg* :header :stamp (ros::time-now))
    (ros::publish "/eus_imitation/robot_state" *robot-state-msg*))


(defun move-robot (&key (av-tm 800))
  (send *ri* :angle-vector (send *pr2* :angle-vector) av-tm))


(defun init-pose ()
  (send *pr2* :reset-pose)
  (move-robot :av-tm 3000)
  (send *ri* :wait-interpolation))

(defun ros-init ()
    ;; init node
    (ros::roseus "eus_imitation")

    ;; publish robot state (and action if data collection mode)
    (ros::advertise "/eus_imitation/robot_state" eus_imitation_msgs::FloatVector 1)
    (unless *imitation-mode*
      (ros::advertise "/eus_imitation/robot_action" eus_imitation_msgs::FloatVector 1))

    ;; subscribe joy (or teleop message)
    (ros::subscribe "/teleop_device/joy" sensor_msgs::Joy #'joy-cb 1)
    (if *imitation-mode*
        (progn
            (ros::ros-info "Imitation control enabled")
            (ros::subscribe "/eus_imitaion/robot_action" eus_imitation_msgs::FloatVector #'(lambda (msg) (setq *robot-action* (send msg :data))) 1)))
  )

(defun joy-cb (msg)
    (setq *joy-msg* msg)
    (setq *joy-buttons* (send msg :buttons)) ;; assume that buttons has 2 elements, for enable (and recording), grasp
    (setq *joy-axes* (send msg :axes)) ;; assume that axes has 6 elements, x, y, z, roll, pitch, yaw

    ;; (setq *trans* (float-vector (subseq *joy-axes* 0 3)))
    ;; (setq *rot* (float-vector (subseq *joy-axes* 3 6)))
    (setq *trans* (subseq *joy-axes* 0 3))
    (setq *rot* (subseq *joy-axes* 3 6))
    (setq *enable-button* (eq (elt *joy-buttons* 0) 1))
    (setq *grasp-button* (eq (elt *joy-buttons* 1) 1))
    (setq *grasp-command* (elt *joy-buttons* 1))


    ;; if not imitation mode, set robot action, x, y, z, roll, pitch, yaw, gripper command
    (unless *imitation-mode*
        (setq *robot-action* (concatenate float-vector *trans* *rot* (float-vector *grasp-command*))))

    (if *enable-button* ;; when enable button is pressed, enable/disable robot and start/stop recording
        (if *enable*
            (progn      ;; if enabled, stop recording and reset robot pose, disable move
                (unix:usleep (* 1000 100))
                (setq *enable* nil)
                (ros::ros-info "disable button pressed, stop recording")
                (ros::wait-for-service "/eus_imitation/rosbag_record_trigger" 1)
                (setq req (instance std_srvs::TriggerRequest :init))
                (setq res (ros::service-call "/eus_imitation/rosbag_record_trigger" req))
                (init-pose)
                )
            (progn      ;; if not enabled, start recording and enable move
                (unix:usleep (* 1000 100))
                (setq *enable* t)
                (ros::ros-info "enable button pressed, recording")
                (ros::wait-for-service "/eus_imitation/rosbag_record_trigger" 1)
                (setq req (instance std_srvs::TriggerRequest :init))
                (setq res (ros::service-call "/eus_imitation/rosbag_record_trigger" req))
                (init-pose)

                ))))

(defun update-action ()
    (if (and (not *imitation-mode*) *enable*) ;; if data collection mode and enabled, publish robot action and move robot
        (progn
            (setq *robot-action-msg* (instance eus_imitation_msgs::FloatVector :init))
            (send *robot-action-msg* :data *robot-action*)
            (send *robot-action-msg* :header :stamp (ros::time-now))
            (ros::publish "/eus_imitation/robot_action" *robot-action-msg*)

            ;; target coords is current end coords + action[:6]
            (setq *target-end-pos* (float-vector
                                    (+ (elt *rarm-end-pos* 0) (elt *robot-action* 0))
                                    (+ (elt *rarm-end-pos* 1) (elt *robot-action* 1))
                                    (+ (elt *rarm-end-pos* 2) (elt *robot-action* 2))))
            (setq *target-end-rpy* (float-vector
                                    (+ (elt *rarm-end-rpy* 0) (elt *robot-action* 3))
                                    (+ (elt *rarm-end-rpy* 1) (elt *robot-action* 4))
                                    (+ (elt *rarm-end-rpy* 2) (elt *robot-action* 5))))
            (setq *target-end-coords* (make-coords :pos *target-end-pos* :rpy *target-end-rpy*))
            (inverse-kinematics *target-end-coords*)
            (setq gripper-command (elt *robot-action* 6))
            (if (> gripper-command 0.5)
                    (send *ri* :start-grasp :rarm :wait nil)
                    (send *ri* :stop-grasp :rarm :wait nil))
            (move-robot))
        (if (and *imitation-mode* *robot-action*) ;; if imitation mode and robot action is, execute robot action
            (progn
                (setq *target-end-pos* (float-vector
                                        (+ (elt *rarm-end-pos* 0) (elt *robot-action* 0))
                                        (+ (elt *rarm-end-pos* 1) (elt *robot-action* 1))
                                        (+ (elt *rarm-end-pos* 2) (elt *robot-action* 2))))
                (setq *target-end-rpy* (float-vector
                                        (+ (elt *rarm-end-rpy* 0) (elt *robot-action* 3))
                                        (+ (elt *rarm-end-rpy* 1) (elt *robot-action* 4))
                                        (+ (elt *rarm-end-rpy* 2) (elt *robot-action* 5))))
                (setq *target-end-coords* (make-coords :pos *target-end-pos* :rpy *target-end-rpy*))
                (inverse-kinematics *target-end-coords*)
                (setq gripper-command (elt *robot-action* 6))
                (if (> gripper-command 0.5)
                        (send *ri* :start-grasp :rarm :wait nil)
                        (send *ri* :stop-grasp :rarm :wait nil))
                (move-robot)))))

(defun main-loop-step ()
    (update-robot)                      ;; update and publish robot state : s_t
    (update-action))                    ;; update, publish and execute robot action : a_t

(defun main-loop ()
    (while (ros::ok)
        (ros::rate *rate*)
        (main-loop-step)
        (ros::spin-once)
        (ros::sleep)))

(defun main (&key (rate 10) (imitation-mode nil))
    (pr2-init)
    (objects (list *pr2*))
    (setq *imitation-mode* imitation-mode)
    (setq *rate* rate)
    (ros-init)
    (update-robot)
    (init-pose)
    (main-loop))
