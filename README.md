# ForecastOptimiser


<p align="center">
  <img src="./targetx/NEF_logo_400x400_light.svg" />
</p>

>This is the implentation of the ForecastOptimiser for [TARGETX](https://target-x.eu)) first Open Call.    
## ⚙ Setup locally

**Host prerequisites**: `docker`, `helm`

After cloning the repository, there are 1 more step to do. 

```bash
cd ForecastOptimiser/helm

# 1.
helm install target-helm targetx-helm -n <your K8s namespace>

```

### Use the application to obtain forecasts 

After the containers are up and running:

 - 
 - access and start playing with the Swager UI at: [localhost:8888/nef/docs](http://localhost:8888/nef/docs)
 - login to the admin dashboard at: [localhost:8888/login](http://localhost:8888/login)
     - Default credentials: `admin` / `targetx`
     - they can be found/changed inside your `.env` file SOS


>\* 💡 Info: *To build all images from their source, you need to run sh create-containers.sh.*

> \*\* 💡 Info: *The shell script used at step 4 (for adding test data) uses `jq` which is a lightweight and flexible command-line JSON processor. You can install it with `apt install jq`*

