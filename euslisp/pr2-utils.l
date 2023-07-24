#!/usr/bin/env roseus
(load "package://informatized_body_core/euslisp/informatized-body-core.l")
(ros::roseus-add-msgs "informatized_body_msgs")
(ros::roseus-add-srvs "std_srvs")
(initialize-ibm-core "pr2_data_augmentation")
(ros::set-logger-level "ros.roseus" 4)

(defun data-collection-start (&key (filename nil))
  (data-collection-start-base "pr2_data_augmentation" :filename filename))

(defun data-collection-stop ()
  (data-collection-stop-base "pr2_data_augmentation"))

(defun update-start (&key (mode 0) (batch -1) (epoch 30) (n_file 1))
  (update-start-base "pr2_data_augmentation" :mode mode :batch batch :epoch epoch :n_file n_file))

(defun recognize-start ()
  (recognize-start-base "pr2_data_augmentation"))

(defun online-update-start (&key (mode 0) (batch -1) (epoch 3) (n_data_thre 10) (n_data_max 30))
  (online-update-start-base "pr2_data_augmentation" :mode mode :batch batch :epoch epoch :n_data_thre n_data_thre :n_data_max n_data_max))

(defun online-update-stop ()
  (online-update-stop-base "pr2_data_augmentation"))

(setq *error-output-obj* *error-output*)
