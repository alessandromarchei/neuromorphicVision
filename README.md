# lis_nv: Visual Navigation Pipeline for Event- and Frame-Based Cameras

This repository contains the ROS Noetic package `lis_nv`, developed for onboard visual navigation on a Raspberry Pi 5. It supports both **event-based cameras (e.g. DVXplorer Micro)** and **frame-based cameras (e.g. FLIR Blackfly using Spinnaker SDK)**, and is designed for real-time deployment alongside PX4 through MAVROS. This code was developed as part of the experiments presented in our paper.

---

## Dependencies

Before building this repository, install the following system and ROS dependencies.

### 🔹 1. Event Camera — `dv-processing`

Install the event-based vision processing library provided by iniVation:

```bash
git clone https://gitlab.com/inivation/dv/dv-processing.git
cd dv-processing
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

> This must be installed **before** building your ROS workspace.


In case another camera is used, you can skip this step. The `dv-processing` library is specifically for event-based cameras like the DVXplorer Micro.
---

### 🔹 2. Frame Camera — Spinnaker Camera Driver

If you're using a FLIR/Blackfly S camera, install the Spinnaker camera driver for ROS:

```bash
cd ~/catkin_ws/src
git clone -b noetic-devel https://github.com/ros-drivers/flir_camera_driver.git
```

In our case, ROS Noetic is used, so we clone the `noetic-devel` branch of the `flir_camera_driver`.
Feel free to use your own branch, available in the `flir_camera_driver` repository.

Then follow the instructions in the `flir_camera_driver` repository to:
- Download and install the Spinnaker SDK from Teledyne FLIR
- Ensure `/opt/spinnaker/include` and `/opt/spinnaker/lib` exist after installation

---

### 🔹 3. MAVROS (PX4 Autopilot)

Install MAVROS and its required geographic dataset:

```bash
sudo apt update
sudo apt install ros-noetic-mavros ros-noetic-mavros-extras

wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
chmod +x install_geographiclib_datasets.sh
sudo ./install_geographiclib_datasets.sh
```

---

## Workspace Setup

This package is intended to be built with `catkin_tools`.

```bash
# Install catkin tools if not already
sudo apt install python3-catkin-tools

# Create your catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Clone this repository
git clone https://github.com/alessandromarchei/lis_nv.git

# (Ensure dv-processing and flir_camera_driver are already installed before building)

# Build the workspace
cd ~/catkin_ws
catkin config --extend /opt/ros/noetic
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build
```


Note that the `CmakeLists.txt` file in this repository is configured to optimize for the Raspberry Pi 5 architecture (ARM). In case you use different architectures, you should change the optimization flags and destination targets accordingly.
---

## Hardware Setup

To build this system, we have used the same fixed-wing drone platform available in this repository: [lis_fixed_wing](https://github.com/lis-epfl/lis-vision-flight).
The drone is equipped with:
- **PX4 Autopilot** (e.g. Pixhawk 4)
- **Raspberry Pi 5** (running Ubuntu 24.04)
- **Event-based camera** (e.g. DVXplorer Micro) or **Frame-based camera** (e.g. FLIR Blackfly S)
- **GPS module** (e.g. u-blox NEO-M8N)
- **Airspeed sensor** (e.g. Sensirion MS4525DO)


## Running the System

Once everything is built, you can launch the appropriate vision node based on the camera you're using:

### For **event-based camera** (e.g. DVXplorer Micro)

```bash
roslaunch lis_nv vision.launch
```

This will:
- Load the configuration from `config/config.yaml`
- Launch MAVROS with PX4 via serial (USB0 at 921600 baud)
- Start the `vision_node` executable

---

### For **frame-based camera** (e.g. FLIR with Spinnaker)

```bash
roslaunch lis_nv vision_frame.launch
```

This will:
- Load the configuration from `config/config_flir.yaml`
- Launch MAVROS with PX4 via serial (USB0 at 921600 baud)
- Start the `vision_node_frames` executable

---

## Collected dataset



## Notes

- Make sure you have permission to access the serial port:
  ```bash
  sudo usermod -aG dialout $USER
  ```
- Reboot or re-login after changing user groups.
- You can check if your PX4 is connected via:
  ```bash
  dmesg | grep tty
  ```
- Configuration files (`config.yaml`, `config_flir.yaml`) can be edited freely to change runtime parameters.

---

## Optional: Run Inside Docker

In our case, we have set up a Docker container to run this entire setup, since we used the Raspberry Pi 5 as the host machine. In fact, ROS1 Noetic was only supported for Ubuntu versions 20.04 and 22.04, and the Raspberry Pi 5 is only supported by Ubuntu 24.04. Therefore, we run this setup inside a Docker container with Ubuntu 22.04 and ROS Noetic.

You can run this whole setup inside a Docker container on Ubuntu 22.04 (hosted on a Raspberry Pi 5) with ROS Noetic. This ensures reproducibility and isolation.

---

## Citation

If you use this code in your research, please cite:


