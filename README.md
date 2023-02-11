### Problem
Implement a fused GELU layer in triton. In pytorch this looks something like this:
```python
def forward(self,
            x: torch.Tensor # [Batch, d_model]
           ) -> torch.Tensor:
    x = self.linear(x) # linear laying mapping d_model to 8 * d_model
    x1, x2 = x.chunk(2, dim=(x.ndim -1))
    return x1 * gelu_fast(x2)

def gelu_fast(x):
    return (
            0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
           )
```
Setup an A100 machine for testing and implement both the forward and backward pass in triton.

### Run
After setup below, run the fused GELU with `./src/fused_gelu.py`

### TODO
- Run benchmark against pytorch to see that the fused kernel offers performance improvement
- Add the bias addition to the fused gelu forward kernel
- The backprop is currently two kernels in serial. Combine them into a single kernel
- The backprop kernel does not concatenation dW1 and dW2 into a single dW and it should

### Setup
- Locally install the gcloud command line interface: `brew install gcloud`, this assumes you already have Homebrew installed on your local Mac
- Create a GCP instance running Ubuntu 20.04 with 1 A100:
```
gcloud compute instances create magic-dev-testing-5 --project=magic-dev-377020 --zone=us-central1-a --machine-type=a2-highgpu-1g --network-interface=network-tier=PREMIUM,subnet=default --maintenance-policy=TERMINATE --provisioning-model=STANDARD --service-account=667948917105-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=count=1,type=nvidia-tesla-a100 --create-disk=auto-delete=yes,boot=yes,device-name=magic-dev-testing-5,image=projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20230125,mode=rw,size=300,type=projects/magic-dev-377020/zones/us-central1-a/diskTypes/pd-ssd --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
```
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
- Cleanup installed packages that are no longer needed: `sudo apt autoremove`
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
- Install triton from source because both the stable and pre-release packages are too old, instructions from [here](https://github.com/openai/triton#install-from-source):
```
git clone https://github.com/openai/triton.git;
cd triton/python;
pip install cmake; # build time dependency
pip install -e .
```
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
matmul-performance:
         M      cuBLAS  cuBLAS (+ torch.nn.LeakyReLU)      Triton  Triton (+ LeakyReLU)
0    256.0    3.640889                       2.730667    3.640889              3.640889
1    384.0   10.053818                       8.507077   11.059200             11.059200
2    512.0   23.831273                      18.724571   21.845333             21.845333
3    640.0   39.384616                      32.000000   39.384616             36.571428
4    768.0   63.195428                      49.151998   52.043293             52.043293
5    896.0   78.051553                      61.083825   73.943582             70.246402
6   1024.0   99.864382                      83.886082   83.886082             80.659693
7   1152.0  135.726544                     110.592000  110.592000            106.642284
8   1280.0  157.538463                     128.000000  136.533337            136.533337
9   1408.0  155.765024                     132.970149  129.804192            126.785488
10  1536.0  181.484314                     153.867127  153.867127            150.593357
11  1664.0  179.978245                     160.694855  163.616581            157.875646
12  1792.0  175.616000                     156.103106  190.498706            187.323738
13  1920.0  209.454544                     186.810803  164.571430            162.635295
14  2048.0  204.600198                     184.365008  184.365008            184.365008
15  2176.0  201.236485                     182.942253  199.244044            195.375226
16  2304.0  248.831991                     225.357284  223.251141            219.154788
17  2432.0  222.971938                     203.583068  197.848332            193.754927
18  2560.0  246.375951                     222.911566  218.453323            214.169933
19  2688.0  219.266228                     201.771569  197.567993            195.531224
20  2816.0  238.329010                     218.071046  216.986107            212.752230
21  2944.0  256.886772                     236.189718  235.075624            231.795505
22  3072.0  239.928400                     222.051395  218.622029            216.118728
23  3200.0  255.999988                     237.037046  235.294114            232.727274
24  3328.0  239.970975                     223.575448  220.832191            218.155436
25  3456.0  258.402470                     239.945145  237.122260            234.365023
26  3584.0  265.237152                     246.343530  224.227912            222.562856
27  3712.0  250.369276                     231.243851  239.561974            237.285850
28  3840.0  250.775504                     233.809729  229.921000            228.024744
29  3968.0  248.521248                     231.544472  244.536938            241.154018
30  4096.0  264.208112                     244.922869  236.715565            233.828792
```
