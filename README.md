# NVIDIA FLare Getting Started
Referenced: https://nvflare.readthedocs.io/en/main/getting_started.html
- Server: Ubuntu 22.04 (docker not necessary)
- Client: Ubuntu 20.04 with docker
- Client on edge: Jetson Nano (Jetpack 4.6.1), Jetson Xavier (Jetpack 5.1), you can use `sudo apt-cache show nvidia-jetpack` to check version.

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
### Build Docker container from Nvidia Base image
reference: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml
(another sourc: https://github.com/dusty-nv/jetson-containers/tree/master/packages/l4t/l4t-ml)

These container have built in ML package for jetson devices, so we can use gpu directly in containers.
(TODO: build docker from base: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base)
* Jetson Nano (Jetpack 4.6.1)
```
sudo docker run -it --runtime=nvidia --name nvflare --ipc=host --ulimit memlock=-1 -p 8102-8103:8102-8103 --ulimit stack=67108864 nvcr.io/nvidia/l4t-base:r32.7.1 bash
```
* Jetson Xavier (Jetpack 5.1)
```
sudo docker run -it --gpus all (or --runtime=nvidia) --name nvflare --ipc=host --ulimit memlock=-1 -p 8102-8103:8102-8103 --ulimit stack=67108864 nvcr.io/nvidia/l4t-ml:r35.2.1-py3 bash
```
If nano install new container, it will not able to use gpu.
### Install packages from requriement.txt
```
apt update
apt-get install git -y
apt-get install --reinstall ca-certificates -y
git clone https://github.com/NVIDIA/NVFlare.git
cd NVFlare
git checkout main
cp -r examples/advanced/cifar10 .
cd cifar10/cifar10-real-world

# You need python(>= 3.8) to install packages
apt install python3.8 -y
apt install libpython3.8-dev -y
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
update-alternatives --set python3 /usr/bin/python3.8
apt install python3-pip -y
python -m pip install --upgrade pip
python -m pip install -U setuptools

#跑這個會跳error: pip install -r ./requirements.txt
```
You will get error due to the tenseal version problem:
```
ERROR: Cannot install nvflare[he]==2.3.0, nvflare[he]==2.3.1 and nvflare[he]==2.3.2 because these package versions have conflicting dependencies.

The conflict is caused by:
    nvflare[he] 2.3.2 depends on tenseal==0.3.12; extra == "he"
    nvflare[he] 2.3.1 depends on tenseal==0.3.12; extra == "he"
    nvflare[he] 2.3.0 depends on tenseal==0.3.12; extra == "he"

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip attempt to solve the dependency conflict

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
```

### Build Tenseal from github
There is no available version on Jetson devices, so you need to build from source:
https://github.com/OpenMined/TenSEAL
```
cd ../../../  (go to the outside from NVFlare)
git clone https://github.com/OpenMined/TenSEAL
cd TenSEAL
apt update
apt install vim -y
vim tenseal/version.py (Edit the version to 0.3.12)
git submodule init
git submodule update
pip install .
(Jetson Nano stucks at here, can't build the wheel for tenseal.)
```
It might encounter some problems due to cmake source(both system and pip have cmake)

`ModuleNotFoundError: No module named 'cmake'`

Make sure you have CMake (3.14 or higher) installed, you can use `cmake --version` to check. If you don't have the required version, you have to update your cmake through:
```
apt remove cmake
apt install cmake
hash -r
cmake --version
```
If the version still not satisfy the required version (In nano, only 3.10.0 can be installed), you need to build from source through:
```
wget https://github.com/Kitware/CMake/releases/download/v3.27.1/cmake-3.27.1.tar.gz
tar -zxvf cmake-3.27.1.tar.gz
cd cmake-3.27.1
apt install libssl-dev -y
./bootstrap
make
make install
bash -r
cmake --version
```
It may take some a considerable time to build the cmake, and Tenseal. 

### Some tips
- These Nvidia container have PyYAML already (not from pip), so you need to remove apt package and install from pip. (try `apt-get purge python3-yaml`)
- The example codes only support `nvidia-smi` to check gpu resources, so it need to be modify (File: /usr/local/lib/python3.8/dist-packages/nvflare/fuel/utils/gpu_utils.py).
```
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
from typing import List


def has_nvidia_smi() -> bool:
    from shutil import which

    # return which("nvidia-smi") is not None
    return True

def use_nvidia_smi(query: str, report_format: str = "csv"):
    if has_nvidia_smi():
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", f"--format={report_format}"],
            capture_output=True,
            text=True,
        )
        rc = result.returncode
        if rc > 0:
            raise Exception(f"Failed to call nvidia-smi with query {query}", result.stderr)
        else:
            return result.stdout.splitlines()
    return None


def _parse_gpu_mem(result: str = None, unit: str = "MiB") -> List:
    gpu_memory = []
    if result:
        for i in result[0:]:
            mem, mem_unit = i.split(" ")
            if mem_unit != unit:
                raise RuntimeError("Memory unit does not match.")
            gpu_memory.append(int(mem))
    return gpu_memory


def get_host_gpu_memory_total(unit="MiB") -> List:
    # result = use_nvidia_smi("memory.total")
    result = "11075 MiB".splitlines()
    return _parse_gpu_mem(result, unit)


def get_host_gpu_memory_free(unit="MiB") -> List:
    # result = use_nvidia_smi("memory.free")
    result = "11075 MiB".splitlines()
    return _parse_gpu_mem(result, unit)


def get_host_gpu_ids() -> List:
    """Gets GPU IDs.

    Note:
        Only supports nvidia-smi now.
    """
    # result = use_nvidia_smi("index")
    gpu_ids = []
    # if result:
    #     for i in result[1:]:
    #         gpu_ids.append(int(i))
    gpu_ids.append(int(0))
    return gpu_ids
```
- l4t-ml:r32.7.1-py3(Jetpack4.6) will have python3.6, which didn't support pip==23.2.1, so need to upgrade python to 3.8 (see below)

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
### Submit jobs
- cifar10_fedavg_stream_tb: Work on Ubuntu, ubuntu docker, xavier docker.
- cifar10_fedavg_he: Work on Ubuntu, ubuntu docker. But arm64 system would encounter below porblem.
```
2023-08-04 06:10:08,630 - HEModelEncryptor - ERROR - Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/nvflare/private/event.py", line 61, in fire_event
    h.handle_event(event, ctx)
  File "/usr/local/lib/python3.8/dist-packages/nvflare/app_opt/he/model_encryptor.py", line 105, in handle_event
    self.tenseal_context = load_tenseal_context_from_workspace(self.tenseal_context_file, fl_ctx)
  File "/usr/local/lib/python3.8/dist-packages/nvflare/app_opt/he/homomorphic_encrypt.py", line 43, in load_tenseal_context_from_workspace
    context = ts.context_from(data)
  File "/usr/local/lib/python3.8/dist-packages/tenseal/__init__.py", line 71, in context_from
    return Context.load(data, n_threads)
  File "/usr/local/lib/python3.8/dist-packages/tenseal/enc_context.py", line 179, in load
    return cls._wrap(ts._ts_cpp.TenSEALContext.deserialize(data))
RuntimeError: incompatible version
```

### Show result
To show the result, download all job results using the `download_job` admin command and specify the download_dir in `./figs/plot_tensorboard_events.py.`
```
client_results_root = "../workspaces/secure_workspace/site-2"
download_dir = "../workspaces/secure_workspace/admin@nvidia.com/transfer"
```
Also, you may need to modify the job in `experiments` in `./figs/plot_tensorboard_events.py.`

- Global_result_Ubuntu Xavier <br> ![Global_result_Ubuntu Xavier](https://github.com/tony0405519/my_NVFlare/assets/32356872/045b668f-2958-473c-97d2-802066f1ecb9)
- Site-1_Xavier_result <br> ![Site-1_Xavier_result](https://github.com/tony0405519/my_NVFlare/assets/32356872/9fdd26e1-8a59-49c7-b09c-4361ec74976d)
- Site-2_Ubuntu_result <br> ![Site-2_Ubuntu_result](https://github.com/tony0405519/my_NVFlare/assets/32356872/bd021cf9-01bc-4aa2-9372-5a0c659be832)




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
