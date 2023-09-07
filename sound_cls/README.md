# Sound Classification Project
- This is the project to use model to do sound classification, which is small than cifar project.
- For dataset, you can download from here: https://github.com/soundata/soundata#quick-example

## Description
- It's a simulation environment for NVFlare architecture with edge devices to do incremental learning. We use one ubuntu host to be server, two ubuntu host and one xavier to be client, 
one TX2 and one nano to be edge device such as camera or microphone.
- The edge device will keep sending data to client for training, and server will start the job every 8 mins.
- After training, the data in client will be remove, and use new data for next training job.
- The server will send best model back to edge device for model inference.
- The final accuracy can reach 56%, almost same as origin training process.

### Client
- In `sound_cls_client`, is the code for training client, put `pt` with project folder.
- `sound_cls/sound_cls_client/pt/learners/sound_learner.py` is the file for training setting, modify the path for dataset.
- Use `sound_cls/sound_cls_client/cifar10-real-world/workspaces/client/site-2/startup/start.sh` to start client, and wait for training.
- Use `sound_cls/sound_cls_client/cifar10-real-world/workspaces/client/site-2/startup/stop_fl.sh` to stop client.

### Server
- Use `sound_cls/sound_cls/mnist-real-world/workspaces/secure_workspace/192.168.100.3/startup/start.sh` to start the server and wait for client's connection.
- Use `sound_cls/sound_cls/mnist-real-world/submit_job.sh` to start a job.
- The config in this folder `sound_cls/sound_cls/mnist-real-world/jobs/mnist_fedavg_stream_tb` can set server and client's behavior.

### Edge
- Use `sound_cls/simulation.sh` to send data to training client.
- Use `sound_cls/socket_server.py` to setup socket server for inference model. NVFlare server will setup a socket connection to edge device when it gets a best model and send to edge device.

- System Architecture <br> ![image](https://github.com/tony0405519/my_NVFlare/assets/32840426/d058092d-9654-4bab-81b3-b940cb0bc7da)
