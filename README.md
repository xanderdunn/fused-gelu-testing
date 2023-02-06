### Setup
- Create a GCP instance running Ubuntu 20.04 with 1 A100:
```
gcloud compute instances create magic-dev-testing-5 --project=magic-dev-377020 --zone=us-central1-a --machine-type=a2-highgpu-1g --network-interface=network-tier=PREMIUM,subnet=default --maintenance-policy=TERMINATE --provisioning-model=STANDARD --service-account=667948917105-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=count=1,type=nvidia-tesla-a100 --create-disk=auto-delete=yes,boot=yes,device-name=magic-dev-testing-5,image=projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20230125,mode=rw,size=300,type=projects/magic-dev-377020/zones/us-central1-a/diskTypes/pd-ssd --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
```
- Locally install the gcloud command line interface: `brew install gcloud`, this assumes you already have Homebrew installed on your local Mac
- Add to ~/.profile: `source "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.bash.inc"`
- SSH into it: `gcloud compute ssh --zone "us-central1-a" "magic-dev-testing-5"  --project "magic-dev-377020"`
- Verify you're on Ubuntu 20.04 Focal: `cat /etc/os-release`
- Update packages: `sudo apt update; sudo apt upgrade`
- Check python version: `python3 --version`: `Python 3.8.10`
- Install NVIDIA Driver and CUDA, instructions from [here](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html). Note that this installs many extraneous packages such as Gnome, so when creating a base image for production usage, prefer to manually install NVIDIA Driver, CUDA, and CUDA Toolkit to avoid the unnecessary bloat:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt install cuda-11-7 --no-install-recommends
```
- In ~/.bashrc:
```
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export PATH=/home/xander/.local/bin:$PATH
```
- Reboot for driver to take effect: `sudo reboot`
- Check A100: `nvidia-smi`:
```
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    45W / 400W |      0MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
- Check CUDA: `ptxas --version`
```
ptxas: NVIDIA (R) Ptx optimizing assembler
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:48:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```
- Install pip3: `sudo apt install -y python3-pip`
- `pip3 install torch`
- Check pytorch: `python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.current_device()); print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"`
```
1.13.1+cu117
True
1
0
NVIDIA A100-SXM4-40GB
11.7
```
- Install triton pre-release: `pip3 install --pre -U triton`, the latest stable is too old.
- Install triton example dependences: `pip3 install numpy matplotlib pandas`
- Check triton
    - Download the example matrix multiplication code from [here](https://triton-lang.org/master/getting-started/tutorials/03-matrix-multiplication.html): `wget https://triton-lang.org/master/_downloads/d5fee5b55a64e47f1b5724ec39adf171/03-matrix-multiplication.py`
    - `python3 03-matrix-multiplication.py`
```
btriton_output=tensor([[-10.9531,  -4.7109,  15.6953,  ..., -28.4062,   4.3320, -26.4219],
        [ 26.8438,  10.0469,  -5.4297,  ..., -11.2969,  -8.5312,  30.7500],
        [-13.2578,  15.8516,  18.0781,  ..., -21.7656,  -8.6406,  10.2031],
        ...,
        [ 40.2812,  18.6094, -25.6094,  ...,  -2.7598,  -3.2441,  41.0000],
        [ -6.1211, -16.8281,   4.4844,  ..., -21.0312,  24.7031,  15.0234],
        [-17.0938, -19.0000,  -0.3831,  ...,  21.5469, -30.2344, -13.2188]],
       device='cuda:0', dtype=torch.float16)
torch_output=tensor([[-10.9531,  -4.7109,  15.6953,  ..., -28.4062,   4.3320, -26.4219],
        [ 26.8438,  10.0469,  -5.4297,  ..., -11.2969,  -8.5312,  30.7500],
        [-13.2578,  15.8516,  18.0781,  ..., -21.7656,  -8.6406,  10.2031],
        ...,
        [ 40.2812,  18.6094, -25.6094,  ...,  -2.7598,  -3.2441,  41.0000],
        [ -6.1211, -16.8281,   4.4844,  ..., -21.0312,  24.7031,  15.0234],
        [-17.0938, -19.0000,  -0.3831,  ...,  21.5469, -30.2344, -13.2188]],
       device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
```
- `sudo apt autoremove`
