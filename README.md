# NVIDIA FLare Getting Started
Referenced: https://nvflare.readthedocs.io/en/main/getting_started.html
- Server: Ubuntu 22.04 (docker not necessary)
- Client: Ubuntu 20.04 with docker
- Client on edge: Jetson Nano (Jetpack 4.6.1), Jetson Xavier (Jetpack 5.1)

## A. Server - Install NVFlare
```
$ python3 -m pip install nvflare
```
Check if nvflare installed successfully
```
$ nvflare
```

Clone NVFLARE repo to get examples, switch main branch (latest stable branch)
```
$ git clone https://github.com/NVIDIA/NVFlare.git
$ cd NVFlare
$ git switch main
```
## B. Client - Containerized Deployment with Docker (Server can skip this part)
### Build nvflare image
Let’s first create a folder called `NVFlare_docker` and then create a file inside named `Dockerfile`:
```
$ mkdir NVFlare_docker
$ cd NVFlare_docker
$ touch Dockerfile
```
Using any text editor to edit the Dockerfile and paste the following:
```
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:23.02-py3
FROM ${PYTORCH_IMAGE}

ARG NVF_VERSION=2.3
ENV NVF_BRANCH=${NVF_VERSION}

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install nvflare

WORKDIR /workspace/
RUN git clone https://github.com/NVIDIA/NVFlare.git --branch ${NVF_BRANCH} --single-branch NVFlare
```

We can then build the new container by running docker build in the directory containing this Dockerfile, for example tagging it nvflare-pt:
```
$ docker build -t nvflare-pt . -f Dockerfile
```
This will result in a docker image, `nvflare-pt:latest`. You can run this container with Docker.

### Docker run with the built image
In this example, a folder `my-workspace` is created and mapped to the folder `my-workspace` in container.
Also, port is also necessary to mounted in the container.
```
$ mkdir my-workspace
$ sudo docker run --rm -it --gpus all\
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ./my-workspace:/my-workspace -p 8102-8103:8102-8103 \
    nvflare-pt:latest
```

## C. Client on Edge - Deploy docker on edge device
### Build Docker container from Nvidia ML image
reference: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml
* Jetson Nano (Jetpack 4.6.1)
```
sudo docker run -it --gpus all --name nvflare --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/l4t-ml:r32.7.1-py3
```
* Jetson Xavier (Jetpack 5.1)
```
sudo docker run -it --gpus all --name nvflare --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/l4t-ml:r35.2.1-py3
```
### Install NVFlare 
```
python3 -m pip install nvflare
```
You will get error: "Tenseal not found" （？）

### Build Tenseal from github
There is no available version on Jetson devices, so you need to build from source:
https://github.com/OpenMined/TenSEAL

Make sure you have CMake (3.14 or higher) installed, you can use `cmake --version` to check. If you don't have the required version, you have to update your cmake through:
```
sudo apt remove cmake
pip3 install cmake
hash -r
cmake --version
```
Then build the Tenseal through:
```
git clone https://github.com/OpenMined/TenSEAL
cd TenSEAL
git submodule init
git submodule update
pip install .
```
It may take some time to build the Tenseal.

# Deploy example project `Real-World Federated Learning with CIFAR-10`
Referenced: https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/cifar10/cifar10-real-world/README.md

The example project from NVFlare suppose that there are 8 clients to run. But in our situation, there are only two hosts, one for server, and the other for client. Therefore, the following tutorial will show how to modify the number of client to 1 for running the example project.

## 1. Install requirements
Install required packages for training, assuming current workpath is: `~/NVFlare/`
```
$ cp -r examples/advanced/cifar10/ .
$ cd cifar10/cifar10-real-world
$ pip install --upgrade pip
$ pip install -r ./requirements.txt
```
Set PYTHONPATH to include custom files of this example:
```
export PYTHONPATH=${PWD}/..
```

## 2. Download the CIFAR-10 dataset
This script will download the CIFAR-10 dataset to `/tmp/cifar10/`:
```
./prepare_data.sh
```

## 3. Create your FL workspace and start FL system
### 3.1 Provisioning 
#### 3.1.1 Modify the project yml file
The project file for creating the secure workspace used in this example is shown in: `./workspaces/secure_project.yml`, but there are some places should be modified.

In participant section, you need to specify the name of server (IP address), client, and admin. You can remove the redundant clients as well (Here we leave 2 clients only).
In builders section, you need to modify the `sp_end_point` to `<IP_address>:8102:8103`.

Notice that `fed_learn_port` and `admin_port` (port: 8102, 8103 by default) should be enabled by firewall and mounted in docker. To enable the port, you can run the following commands:
```
$ sudo ufw allow from <source IP> to any port 8102
$ sudo ufw allow from <source IP> to any port 8103
```
#### 3.1.2 Create the secure workspace
```
cd ./workspaces
nvflare provision -p ./secure_project.yml
cp -r ./workspace/secure_project/prod_00 ./secure_workspace
cd ..
```

### 3.2 Multi-tasking resource management
In this example, we assume N_GPU local GPUs, each with at least 8 GB of memory, are available on the host system. To find the available number of GPUs, run
```
export N_GPU=$(nvidia-smi --list-gpus | wc -l)
echo "There are ${N_GPU} GPUs available."
```

We can change the clients' local GPUResourceManager configurations to show the available N_GPU GPUs at each client.

Each client needs about 1 GB of GPU memory to run an FL experiment with the CIFAR-10 dataset. Therefore, each client needs to request 1 GB of memory such that 8 (2 in our case) can run in parallel on the same GPU.

To request the GPU memory, set the "mem_per_gpu_in_GiB" value in the job's meta.json file.

To update the clients' available resources, we copy resource.json.default to resources.json and modify them as follows:
```
n_clients=2
for id in $(eval echo "{1..$n_clients}") 
do
  client_local_dir=workspaces/secure_workspace/site-${id}/local 
  cp ${client_local_dir}/resources.json.default ${client_local_dir}/resources.json
  sed -i "s|\"num_of_gpus\": 0|\"num_of_gpus\": ${N_GPU}|g" ${client_local_dir}/resources.json
  sed -i "s|\"mem_per_gpu_in_GiB\": 0|\"mem_per_gpu_in_GiB\": 1|g" ${client_local_dir}/resources.json 
done
```

### 3.3 Start FL system
By far, the folder structure is as followed:
```
~/NVFlare/
 ...
 |--cifar10/
    ...
    |--cifar10-real-world/
       ...
       |--jobs/
          |--cifar10_fedavg_he/ (job's name)
             |--meta.json
             |--cifar10_fedavg_he/config/
                |--config_fed_client.json
                |--config_fed_server.json
          |--cifar10_fedavg_stream_tb/
             ...(structure same as cifar10_fedavg_he)
       |--workspaces/
             |--secure_project.yml
             |--workspace/
             |--secure_workspace/
                |--192.168.100.3/ (server's name)
                   ...
                   |--startup/
                      ...
                      |--start.sh
                      |--stop_fl.sh
                      |--sub_start.sh
                |--site-1/
                   ...
                   |--startup/
                      ...
                      |--start.sh
                      |--stop_fl.sh
                      |--sub_start.sh
                |--site-2/
                ...
       |--start_fl_secure.sh
       |--submit_job.py
       |--submit_job.sh
```
#### 3.3.1 Forward client's folder
You should forward the `site-?` folder to the client's host.

#### 3.3.2 Start the server
In the example project, admin will devide Cifar10 dataset into 8 sub-datasets for each client, and the datapath is specified.

In our case, we hope to specify the datapath by client and use the whole dataset for training because we only have one client. Therefore, some config files should be modified.

1. In file `jobs/cifar10_fedavg_he/meta.json`, `min_client` should be set to 1.

2. In file `jobs/cifar10_fedavg_he/cifar10_fedavg_he/config/config_fed_client.json`, `TRAIN_SPLIT_ROOT` should not be set.

3. In file `jobs/cifar10_fedavg_he/cifar10_fedavg_he/config/config_fed_server.json`, the component `data_splitter` can be deleted. (But in our example, because we didn't implement our own dataloader, we still need to use spliter) -> send /tmp/cifar10_splits/${job_id}/site-1.npy to client 'site-1'. (This file will appear after submit_job)

For starting the `server` of FL system in the secure workspace, run
```
cd ~/NVFlare/cifar10/cifar10-real-world/
export PYTHONPATH="${PYTHONPATH}:${PWD}/.."
echo "PYTHONPATH is ${PYTHONPATH}"
./workspaces/secure_workspace/192.168.100.3/startup/start.sh
```

#### 3.3.3 Start the client
Take `site-1` as an example
```
cd ~/NVFlare/cifar10/cifar10-real-world/
./prepare_data.sh
```
In this example, we need to use splited data from admin.
So move ./site-1.npy(which is from server) to /tmp/cifar-10/
And set dataset path in ~/NVFlare/cifar10/pt/learners/cifar10_learner.py
Then start client:
```
export PYTHONPATH="${PYTHONPATH}:${PWD}/.."
echo "PYTHONPATH is ${PYTHONPATH}"
./workspaces/secure_workspace/site-1/startup/start.sh
```


#### 3.3.4 Submit the job
Admin submits the job by running:
```
./submit_job.sh cifar10_fedavg_he 1.0
```
or 
```
./submit_job.sh cifar10_fedavg_stream_tb 1.0
```
It will call APIRunner to submit job.

#### 3.3.5 Monitor/Stop the FL system
You can monitor the system through NVIDA admin API by running:
```
./workspaces/secure_workspace/admin@nvidia.com/startup/fl_admin.sh
```
Enter the admin's name:
```
User Name: admin@nvidia.com
```
Type `?` to show usage of a command, here are some common commands to used:
```
list_jobs
check_status client
check_status server
abort_job <job_id>
delete_job <job_id>
shutdown client
shutdown server
```

---
### NVIDIA FLARE Workspace (Ignore below) 
In folder: 'cifar10/cifar10-real-world'

submit_job後, 資料夾結構如下：
```
prepare_data.sh
README.txt
requirements.txt
run_experiments.sh
shutdown_fl_run.sh
start_fl_secure.sh
submit_job.py
submit_job.sh
/figs/
/jobs/
    cifar10_fedavg_he/
        cifar10_fedavg_he
        meta.json
    cifar10_fedavg_stream_tb/ 
/workspaces/
    secure_project.yml
    secure_workspace/
        192.168.100.3/
            audit.log
            log.txt
            local/
            
            startup/
                authorization.json
                fed_server.json
                log.config
                readme.txt
                rootCA.pem
                server_context.tenseal
                server.crt
                server.key
                signature.pkl
                start.sh
                stop_fl.sh
                sub_start.sh
            transfer/
            068e85c5-19ff-45c1-84db-6a2ccd079df8/
                app_server/
                    ...
                    config/
                        config_fed_server.json
                fl_app.txt
                log.txt

        site-1/
            log.txt
            startup/
                client_context.tenseal
                client.crt
                client.key
                fed_client.json
                log.config
                readme.txt
                rootCA.pem
                signature.pkl
                start.sh
                stop_fl.sh
                sub_start.sh
            transfer/
            068e85c5-19ff-45c1-84db-6a2ccd079df8/
                app_site-1/
                    best_local_model.pt
                    local_model.pt
                    config/
                        config_fed_client.json
                        config_fed_server.json
                fl_app.txt
                log.txt
                meta.json
                tb_events/
                    site-1/events.out.tfevents.1690776735.aienode2.373638.0
            
        admin@nvidia.com/
            local/
            startup/
                client.crt
                client.key
                client.pfx
                f1_admin.sh
                fed_admin.json
                rootCA.pem
                readme.txt
            transfer/


```
