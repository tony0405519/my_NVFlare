# NVIDIA FLARE Docker run

# Prepare Environment for cifar10
https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/cifar10/cifar10-real-world/README.md

# NVIDIA FLARE Workspace 
In folder: 'cifar10-cifar10-real-world'

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
