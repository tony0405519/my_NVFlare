 # NVIDIA FLare Getting Started
Referenced: https://nvflare.readthedocs.io/en/main/getting_started.html
- Server: Ubuntu 22.04 (docker not necessary)
- Client: Ubuntu 20.04 with docker

## Install NVFlare
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
## Containerized Deployment with Docker (Server can skip this part)
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
    -v ./my-workspace/:/my-workspace -p 8000-9000:8000-9000 \
    nvflare-pt:latest
```

# Deploy example project `Real-World Federated Learning with CIFAR-10`
Referenced: https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/cifar10/cifar10-real-world/README.md
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
### Provisioning 
The project file for creating the secure workspace used in this example is shown in: `./workspaces/secure_project.yml`, but there are some places should be modified.

In participant section, you need to specify the name of server (IP address), client, and admin. You can remove the redundant clients as well.
In builders section, you need to modify the `sp_end_point` to <IP_address>:8102:8103 

Notice that `fed_learn_port` and `fed_learn_port` (port: 8102, 8103 by default) should be enabled by firewall and mounted in docker. To enable the port, you can run the following commands:
```
$ sudo ufw allow from <source IP> to any port 8102
$ sudo ufw allow from <source IP> to any port 8103
```


# NVIDIA FLARE Workspace 
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
