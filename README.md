# How to build a  Chatbot  Llama 2 on  the EC2 Spot Instance.   

## Introduction

Llama 2 Large Language Model (LLM) is a successor to the Llama 1 model released by Meta. Primarily, Llama 2 models are available in three model flavors that depending on their parameter scale range from 7 billion to 70 billion, these are `Llama-2-7b`, `Llama-2-13b`, and `Llama-2-70b`. Llama 2 LLM models have a commercial, and open-source license for research and non-commercial use.eef

This article explains how to use the Meta Llama 2 large language model (LLM) on a Cloud GPU server. 

In ordering to run our LLM we require a computer with GPU,  LLMs with higher nuber of paramters requires higher GPU memory.

| **Instance Size** | **GPU** | **vCPUs** | **Memory (GiB)** | **Instance Storage (GB)** | **Network Bandwidth (Gbps)** | **EBS Bandwidth (Gbps)** | **On-Demand Price/hr\*** | **1-yr Reserved Instance Effective Hourly\* (Linux)** | **3-yr Reserved Instance Effective Hourly\* (Linux)** |      |
| ----------------- | ------- | --------- | ---------------- | ------------------------- | ---------------------------- | ------------------------ | ------------------------ | ----------------------------------------------------- | ----------------------------------------------------- | ---- |
| g4dn.xlarge       | 1       | 4         | 16               | 1 x 125 NVMe SSD          | Up to 25                     | Up to 3.5                | $0.526                   | $0.316                                                | $0.210                                                |      |
| g4dn.2xlarge      | 1       | 8         | 32               | 1 x 225 NVMe SSD          | Up to 25                     | Up to 3.5                | $0.752                   | $0.452                                                | $0.300                                                |      |
| g4dn.4xlarge      | 1       | 16        | 64               | 1 x 225 NVMe SSD          | Up to 25                     | 4.75                     | $1.204                   | $0.722                                                | $0.482                                                |      |
| g4dn.8xlarge      | 1       | 32        | 128              | 1 x 900 NVMe SSD          | 50                           | 9.5                      | $2.176                   | $1.306                                                | $0.870                                                |      |
| g4dn.16xlarge     | 1       | 64        | 256              | 1 x 900 NVMe SSD          | 50                           | 9.5                      | $4.352                   | $2.612                                                | $1.740                                                |      |
|                   |         |           |                  |                           |                              |                          |                          |                                                       |                                                       |      |
|                   |         |           |                  |                           |                              |                          |                          |                                                       |                                                       |      |

\* Prices shown are for US East (Northern Virginia) AWS Region. Prices for 1-year and 3-year reserved instances are for "Partial Upfront" payment options or "No Upfront" for instances without the Partial Upfront option.

## Access the Llama 2 LLM Model

In this section, configure your HuggingFace account to access and download the Llama 2 family of models.

1. Request access to Llama2 through the [official Meta downloads page](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

   ![Meta Llama2 access page](assets/images/posts/README/GW6V2RI.png)

   When prompted, enter the same email address as your HuggingFace account, and wait for a Meta confirmation email.

2. Scroll down on the page, check the terms and conditions box, then click **Accept and Continue** to continue.

3. Log in to your  [HuggingFace account](https://huggingface.co/join), and navigate to settings

   

4. On the left navigation menu, click **Access Tokens**.

   ![Hugging Face Access Tokens menu option](assets/images/posts/README/qXyaaRa.png)

5. Click the **New token** button to set up a new access token.

6. Give the token a name for example: `meta-llama`, set the role to **read**, and click the **Generate a Token** button to save.

   ![Generate a new Hugging Face Access Token](assets/images/posts/README/5jBvxYB.png)

7. Click the **Show** option to reveal your token in plain text. Copy the token to your clipboard.

8. In your Hugging Face interface, enter `Llama-2-7b` in the search bar to open the model page.

9. Click the checkbox to share your information with Meta, and click **Submit** to request access to the model repository.

   ![Access the Meta Llama 2 Hugging Face repository](assets/images/posts/README/dVgwu1B.png)

When successful, you should receive a confirmation email from HuggingFace accepting your request to access the model. This confirms that you can use the model files as permitted by the Meta terms and conditions.

# Create EC2 Instance with GPU

First we need to login to our AWS Account

[AWS Account](https://console.aws.amazon.com/console/home?nc2=h_ct&src=header-signin)

Then go to your menu and click Services Quotas

![image-20230919204207495](assets/images/posts/README/image-20230919204207495.png)

and select `All G and VT Spot Instances Requests` and click **Request increase account level**  and choose **4** otherwise wont towk. Because the cheaper GPU currently on AWS has 4 vcps and if yo request 1, you wont launch the instance. 

![image-20230919220441109](assets/images/posts/README/image-20230919220441109.png)

After few hours maximum on day, will be available the spot instances for P4 instances. In ordert 

If you want use on demand you can use `Running on Demand G and VT instances.`  and agiain you request 4.  

![image-20230920215755890](assets/images/posts/README/image-20230920215755890.png)



## Creation of the Security Group

Since our Chatbot will use **Streamlit**,  normally runs on port **8501** or **850*** we will have to create a new firewall run to allow a custom port for our streamlit app.

On the menu bar we find **Security Groups**

![r](assets/images/posts/README/image-20230919211023354.png)

We open the Feature of EC2 and we click 	 **Create Security Group**

 So at the configure security group we will a custom tcp port by clicking on the **ADD RULE.** Then select **Custom TCP Port .** In the port range you will then change it to **8501**, **8888**, and **22**. Finally you will select **anywhere** in the sources section. and  for outbound keep the default values and click create security group.

![image-20230919211523944](assets/images/posts/README/image-20230919211523944.png)

For production for sure you can remove the SSH and the Jupyter Notebook ports.

## Launch EC2 Instance

We go to EC2 and we create an instance **Chatbot** and we choose **Ubuntu 22.04** with server **g4dn.xlarge**

![image-20231002214752315](assets/images/posts/README/image-20231002214752315.png)



we create a private key, **you download this key**  because we will  use later  and for security group select existing security group ,  **Chatbot with Streamlit**,

![image-20231002214902234](assets/images/posts/README/image-20231002214902234.png)

Then we require to add additional storage space to our Virtual Instance, we choose 100gb

![image-20231002215129289](assets/images/posts/README/image-20231002215129289.png)



If we want to save money , in **Advanced details** we must put **spot.** 

![image-20230919214933335](assets/images/posts/README/image-20230919214933335.png)

and we keep all the reamaining settings as default and click launch

![image-20231002215435058](assets/images/posts/README/image-20231002215435058.png)

Be **aware** that now time is money. So **do not forget to delete your instance** after you finish this demo.

Now the Instance is initializing , we wait few minutes and then we click on **Connect** 

​	![image-20231002220022923](assets/images/posts/README/image-20231002220022923.png)

then we select the **SSH Client**

![image-20231002220110688](assets/images/posts/README/image-20231002220110688.png)



Then we have to open our terminal, then go where you downloaded your private key

1. Run this command, if necessary, to ensure your key is not publicly viewable.
2. ```
    chmod 400 MyPrivateKey.pem
   ```

3. Connect to your instance using its Public DNS for example:

```
 ssh -i "MyPrivateKey.pem" ubuntu@ec2-3-234-217-147.compute-1.amazonaws.com
```

and you will got something like

![image-20231002220549481](assets/images/posts/README/image-20231002220549481.png)

# Python Installation

Add the `deadsnakes` PPA repository to the system next. The simplest method for installing Python 3.10 is as follows:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
```

Use this command to install Python 3.10 

```bash
sudo apt install python3.10 -y
```

we click in OK	

![image-20231002220705001](assets/images/posts/README/image-20231002220705001.png)

​	we check the Python Version

```bash
python3 --version
```

![image-20231002220753015](assets/images/posts/README/image-20231002220753015.png)

then we install our friend pip

```
sudo apt install python3-pip -y
```

![image-20231002220853783](assets/images/posts/README/image-20231002220853783.png)

# Install CUDA & cuDNN 

### Update & upgrade

```
sudo apt update && sudo apt upgrade -y
```

Just in case you want to remove previous NVIDIA installation you can do `sudo apt autoremove nvidia* --purge`, but we skip.

```
sudo apt install ubuntu-drivers-common -y
```



## Check Ubuntu devices

```
ubuntu-drivers devices
```

![image-20231002221249412](assets/images/posts/README/image-20231002221249412.png)

You will install the NVIDIA driver whose version is tagged with **recommended**

## Install Ubuntu drivers

```
sudo ubuntu-drivers autoinstall
```

### Install NVIDIA drivers

My **recommended** version is 525, adapt to yours

```
sudo apt install nvidia-driver-525 -y
```



![image-20231002221922525](assets/images/posts/README/image-20231002221922525.png)

### Reboot & Check

IF you were not on the cloud you can simply type `reboot` but here we have to reboot our instanced manually

```
exit
```

![image-20231002222204934](assets/images/posts/README/image-20231002222204934.png)

after that we reconnect via ssh  verify that the following command works

```
nvidia-smi
```

and wualla



![image-20231002222327748](assets/images/posts/README/image-20231002222327748.png)

 that is <img src="assets/images/posts/README/image-20231002222952319.png" alt="image-20231002222952319" style="zoom:1000%;" />

## Install CUDA drivers

### Install CUDA toolkit

```
sudo apt install nvidia-cuda-toolkit -y
```

### Check CUDA install

```
nvcc --version
```



![image-20231002223426097](assets/images/posts/README/image-20231002223426097.png)



## Install cuDNN

### Download cuDNN .deb file

You can download cuDNN file [here](https://developer.nvidia.com/rdp/cudnn-download). You will need an Nvidia account. Select the cuDNN version for the appropriate CUDA version, which is the version that appears when you run:

```
nvcc --version
```

![image-20231002223757709](assets/images/posts/README/image-20231002223757709.png)





## Transferring Files Using SCP 

Secure Copy Protocol (SCP) is a means of securely transferring computer files between a local and a remote host or between two remote hosts. It’s based on the Secure Shell (SSH) protocol.

To copy a file from your local system to your EC2 instance, use the following command:

```bash
scp -i "MyPrivateKey.pem" C:\Users\Downloads\cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb  ubuntu@ec2-3-234-217-147.compute-1.amazonaws.com:/home/ubuntu

```

and you can check if was uploaded

![image-20231002230100421](assets/images/posts/README/image-20231002230100421.png)

### Install cuDNN

```
sudo apt install ./cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb

```

![image-20231002230324775](assets/images/posts/README/image-20231002230324775.png)

```
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.5.29/cudnn-local-535C49CB-keyring.gpg /usr/share/keyrings/
```

My cuDNN version is 8, adapt the following to your version:

```
sudo apt update
sudo apt install libcudnn8
sudo apt install libcudnn8-dev
sudo apt install libcudnn8-samples
```

![image-20231002230822442](assets/images/posts/README/image-20231002230822442.png)



## Test CUDA on Pytorch

### Create a virtualenv and activate it

```
sudo apt-get install python3-pip
sudo pip3 install virtualenv 
virtualenv -p py3.10 venv
source venv/bin/activate
```

![image-20231002231032899](assets/images/posts/README/image-20231002231032899.png)

### Install pytorch

```
pip3 install torch torchvision torchaudio
```

![image-20231002231403794](assets/images/posts/README/image-20231002231403794.png)



### Open Python and execute a test



```
import torch
print(torch.cuda.is_available()) # should be True
```

![image-20231002231554867](assets/images/posts/README/image-20231002231554867.png)

```
t = torch.rand(10, 10).cuda()
print(t.device) # should be CUDA
```

![image-20231002231624969](assets/images/posts/README/image-20231002231624969.png)

# Create an AMI from an Amazon EC2 Instance

In ordering to save our EC2 Instance Setup that we have done before  got to **Amazon EC2 Instances** view, you can create Amazon Machine Images (AMIs) from either running or stopped instances. 

*To create an AMI from an instance*

1. Right-click the instance you want to use as the basis for your AMI, and choose **Create Image** from the context menu.

   ![image-20231002231900549](assets/images/posts/README/image-20231002231900549.png)

   **Create Image** context menu

2. In the **Create Image** dialog box, type a unique name and description, and then choose **Create Image**. By default, Amazon EC2 shuts down the instance, takes snapshots of any attached volumes, creates and registers the AMI, and then reboots the instance. Choose **No reboot**if you don't want your instance to be shut down.

   

   ###### Warning

   If you choose **No reboot**, we can't guarantee the file system integrity of the created image.

   ![image-20231002232421856](assets/images/posts/README/image-20231002232421856.png)

   **Create Image** dialog box

It may take a few minutes for the AMI to be created. After it is created, it will appear in the **AMIs** view in AWS Explorer. To display this view, double-click the **Amazon EC2 | AMIs** node in AWS Explorer. To see your AMIs, from the **Viewing** drop-down list, choose **Owned By Me**. You may need to choose **Refresh** to see your AMI. When the AMI first appears, it may be in a pending state, but after a few moments, it transitions to an available state.

## Install Model Dependencies

To use the model features and tools, install Jupyter Notebook to run commands, then install the required libraries as described in the steps below.

1. Install PyTorch.

   ```
   pip3 install torch --index-url https://download.pytorch.org/whl/cu118
   ```

   The above command installs the PyTorch library that offers efficient tensor computations and supports GPU acceleration for training operations.

   To install a PyTorch version that matches your CUDA visit the [documentation page](https://pytorch.org/get-started/locally/) to set preferences and run the install command.

2. Install dependency packages.

   ```
   pip3 install bitsandbytes scipy transformers accelerate einops xformers
   ```

   Below is what each package represents:

   - `transformers`: It's used for Natural Language Processing (NLP) tasks, and key functionalities include tokenization and fine tuning.
   - `accelerate`: Improves the training and inference of machine learning models.
   - `einops`: Reshapes and reduces the dimensions of multi-dimensional arrays.
   - `xformers`: Provides multiple building blocks for making transformer-based models.
   - `bitsandbytes`: Focuses on functions that optimize operations involving 8-bit data, such as matrix multiplication.
   - `scipy`: Enables access to the `bitsandbytes` functionalities for scientific, and technical computing.

3. Install the Jupyter `notebook` package.

   ```
   pip3 install notebook
   sudo -H pip install jupyter
   ```
   
4. Allow incoming connections to the Jupyter Notebook port `8888`.

   ```
   sudo ufw allow 8888
   ```

5.  We copy the public ip

   ```
   curl http://checkip.amazonaws.com
   ```

6. Start Jupyter Notebook.

   ```
   jupyter notebook --ip=0.0.0.0
   ```

   If you receive the following error:

   ```
   Command 'jupyter' not found, but can be installed with:
   ```

   End your SSH connection, and reconnect to the server to refresh the cache.

   When successful, Jupyter Notebook should start with the following output:

   ```
   [I 2023-07-31 00:29:42.997 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
   [W 2023-07-31 00:29:42.999 ServerApp] No web browser found: Error('could not locate runnable browser').
   [C 2023-07-31 00:29:42.999 ServerApp] 
       To access the server, open this file in a browser:
           file:///home/example-user/.local/share/jupyter/runtime/jpserver-69912-open.html
       Or copy and paste one of these URLs:
           http://HOSTNAME:8888/tree?token=e536707fcc573e0f19be40d90902825ec6e04181bed85be9
           http://127.0.0.1:8888/tree?token=e536707fcc573e0f19be40d90902825ec6e04181bed85be9
   ```

   As displayed in the above output, copy the generated token URL to securely access Jupyter Notebook in your browser.

7. In a web browser such as Chrome, access Jupyter Notebook using your generated access token.

   ```
   http://SERVER-IP:8888/tree?token=YOUR=TOKEN
   ```

 http://54.152.53.77:8888/tree?token=7772646a5c9f452aa1e8e9b8ad293f3fbb4765159f3f6829

```
pip3 install ipykernel notebook

```

```
python3 -m ipykernel install --user --name GPT --display-name "Python (GPT)"

```

## Run Llama 2 70B Model

In this section, initialize the `Llama-2-70b-hf` model in 4-bit and 16-bit precision, and add your Hugging Face authorization key to initialize the model pipeline and tokenizer as described in the steps below.

1. Access the Jupyter Notebook web interface.

2. On the top right bar, click **New** to reveal a dropdown list.

   ![Create a new Jupyter Notebook](assets/images/posts/README/vksiEEV.jpg)

3. Click **Notebook**, and select `Python 3 (ipykernel)` to open a new file.

4. In the new Kernel file, click the filename, by default, it's set to `Untitled`.

5. Rename the file to `Llama-2-70b`, and press :key:Enter: to save the new filename.

   ![Rename Jupyter Notebook file](assets/images/posts/README/j9VPuZm.png)

6. In a new code cell, initialize the `Llama-2-70b-hf` model.

   ```
   from torch import cuda, bfloat16
   import transformers
   #model_id = 'meta-llama/Llama-2-70b-hf'
   model_id = 'Stevross/Astrid-LLama-3B-CPU'
   device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
   quant_config = transformers.BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type='nf4',
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=bfloat16
   )
   auth_token = 'YOUR_AUTHORIZATION_TOKEN'
   model_config = transformers.AutoConfig.from_pretrained(
       model_id,
       use_auth_token=auth_token
   )
   model = transformers.AutoModelForCausalLM.from_pretrained(
       model_id,
       trust_remote_code=True,
       config=model_config,
       quantization_config=quant_config,
       use_auth_token=auth_token
   )
   model.eval()
   print(f"Model loaded on {device}")
   ```
   
   Paste your Hugging Face token next to the `auth_token =` directive to replace `YOUR-AUTHORIZATION_TOKEN`.
   
   The above code sets the `model_id` and enables 4-bit quantization with `bitsandbytes`. This applies 4-bit to less relevant parts of the model and 16-bit quantization to the text-generation parts of the model. In 16-bit, the output is less degraded providing near-accurate information.
   
7. Click the play button on the top menu bar, or press CTRL + ENTER to run the initialize the model.

   When successful, the code prints the device it runs on, and shows the model is successfully downloaded. The download process may take about 30 minutes to complete.

8. In a new code cell, initialize the tokenizer.

   ```
   tokenizer = transformers.AutoTokenizer.from_pretrained(
       model_id,
       use_auth_token=auth_token
   )
   ```
   
   The above code sets the tokenizer to `model_id`. Every LLM has a different tokenizer that converts text streams to smaller units for the language model to understand and interpret the input.
   
9. Initialize the pipeline.

   ```
   pipe = transformers.pipeline(
       model=model, 
       tokenizer=tokenizer,
       task='text-generation',
       temperature=0.0, 
       max_new_tokens=50,  
       repetition_penalty=1.1 
   )
   ```
   
   The above code initializes the pipeline for text generation through which you can manipulate the kind of response to generate using the model. To enhance the output, the pipeline accepts additional parameters.
   
10. Run the following code to add a text prompt to the pipeline. Replace `Hello World` with your desired prompt.

    ```
    result = pipe('Hello World')[0]['generated_text']
    
    print(result)
    ```

    The above code block generates output based on the input prompt. To generate a response, it can take up to 5 minutes to complete.

11. Verify the GPU usage statistics.

    ```
    !nvidia-smi
    ```

    Output:

    ```
    +-----------------------------------------------------------------------------+
    
    |  Processes:                                                                 |
    
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    
    |        ID   ID                                                   Usage      |
    
    |=============================================================================|
    
    |    0    0    0      35554      C   /usr/bin/python3                37666MiB |
    
    +-----------------------------------------------------------------------------+
    ```

    As displayed in the above output, the `Llama-2-7b-hf` model uses `37.6 GB` of GPU memory when executed with 4-bit precision and quantization. In full precision, the model VRAM consumption is much higher.

## Run the Llama 2 70B Chat Model

In this section, initialize the `Llama-2-70b-chat-hf` fine-tuned model with 4-bit and 16-bit precision as described in the following steps.

1. On the main menu bar, click **Kernel**, and select **Restart and Clear Outputs of All Cells** to free up the GPU memory.

   ![Free GPU Memory in Jupyter Notebook](assets/images/posts/README/lpj1FyM.png)

2. Click **File**, select the **New** dropdown, and create a new **Notebook**.

3. Rename the notebook to `Llama-2-7b-chat-hf`.

4. Initialize the `Llama-2-70b-chat-hf` model. Replace `AUTHORIZATION_TOKEN` with your Hugging Face access token on the `auth_token =` directive.

   ```
   from torch import cuda, bfloat16
   import transformers
   model_id = 'meta-llama/Llama-2-70b-chat-hf'
   model_id = 'Stevross/Astrid-LLama-3B-CPU'
   device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
   quant_config = transformers.BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type='nf4',
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=bfloat16
   )
   auth_token = 'YOUR_AUTHORIZATION_TOKEN'
   model_config = transformers.AutoConfig.from_pretrained(
       model_id,
       use_auth_token=auth_token
   )
   model = transformers.AutoModelForCausalLM.from_pretrained(
       model_id,
       trust_remote_code=True,
       config=model_config,
       quantization_config=quant_config,
       use_auth_token=auth_token
   )
   model.eval()
   print(f"Model loaded on {device}")
   ```

   The above code uses the fine-tuned chat model `Llama-2-7b-chat-hf`, and your access token to access the model.

5. Click the play button, or press CTRL + ENTER to execute the code.

6. Initialize the tokenizer.

   ```
   tokenizer = transformers.AutoTokenizer.from_pretrained(
       model_id,
       use_auth_token=auth_token
   )
   ```
   
7. Initialize the pipeline.

   ```
   pipe = transformers.pipeline(
       model=model, 
       tokenizer=tokenizer,
       task='text-generation',
       temperature=0.0, 
       max_new_tokens=50,  
       repetition_penalty=1.1
   )
   ```
   
8. Add a text prompt to the pipeline. Replace `Hello World` with your desired prompt.

   ```
   result = pipe('Hello World')[0]['generated_text']
   print(result)
   ```
   
   In the chat model, the prompt you enter must be in a dialogue format to differentiate the responses between the base model and the fine-tuned version.
   
9. Verify the GPU usage statistics.

   ```
   !nvidia-smi
   ```

   Output:

   ```
   +-----------------------------------------------------------------------------+
   
   | Processes:                                                                  |
   
   |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
   
   |        ID   ID                                                   Usage      |
   
   |=============================================================================|
   
   |    0    0    0      36099      C   /usr/bin/python3                37666MiB |
   
   +-----------------------------------------------------------------------------+
   ```

   As displayed in the above output, the `Llama-2-70b-hf` model uses up to `37.6 GB` of VRAM when executed with 4-bit precision and quantization. The VRAM consumption of both the base model and fine-tuned models is similar because it's directly proportional to the parameter range of 70 billion.

## Llama 2 Model Weights

Llama 2 parameters range from 7 billion to 70 billion, and each model has a fine-tuned chat version. Models with a low parameter range consume less GPU memory and can apply to testing inference on the model with fewer resources, but with a tradeoff on the output quality.

The following model options are available for Llama 2:

- `Llama-2-13b-hf`: Has a 13 billion parameter range and uses `8.9 GB` VRAM when run with 4-bit quantized precision.
- `Llama-2-13b-chat-hf`: A fine-tuned version of the 13 billion base model designed to have Chatbot-like functionality.
- `Llama-2-7b-hf`: Has a 7 billion parameter range and uses `5.5 GB` VRAM when executed with 4-bit quantized precision.
- `Llama-2-7b-chat-hf`: A fine-tuned version of the 7 billion base model. The VRAM consumption matches the base model and works like a chatbot.

The above models are open-source and commercially licensed, you can use them for research and commercial purposes.

## Llama 2 improvements over Llama 1

Llama 2 has significant advantages over its predecessor Llama 1 with more variants available on both the base and fine-tuned version.

1. Unlike Llama 1, Llama 2 is open-sourced and commercially available to use.
2. Llama 2 has a parameter range of 7 to 70 billion while Llama 1 has a parameter range of 7 to 65 billion.
3. The Llama 2 model trains on 2 trillion tokens which is 40% more tokens than Llama 1. This increases its accuracy and knowledge in outputs.
4. Llama 2 has a context length of 4096 which is double the context length of Llama 1.
5. Llama 2 offers better results against standard benchmarks such as World Knowledge, Reading Comprehension, and Commonsense Reasoning as compared to Llama 1.
6. Llama 2 offers fine tuned chat models together with base models while Llama 1 only offers base models.

## Common Declarations

1. `trust_remote_code`: Assesses code trustworthiness, integrity, and safety measures based on its origin when fetching code from external sources.
2. `task`: Sets the pipeline task to text generation.
3. `temperature`: With a maximum value of 1.0 and a minimum value of 0.1, it controls the output randomness. Higher values (closer to 1.0) lead to more randomness in the output.
4. `max_new_tokens`: Defines the number of tokens in the output. If not defined, the model produces an output with a random number of tokens.
5. `repetition_penalty`: Manages the likelihood of generating repeated tokens. Higher values reduce the occurrence of repeated tokens, and vice versa.

## Conclusion

In this article, you used Meta Llama 2 models on a Vultr Cloud GPU Server, and run the latest Llama 2 70b model together with its fine-tuned chat version in 4-bit mode. Below are the VRAM usage statistics for Llama 2 models with a 4-bit quantized configuration on an 80 GB RAM A100 Vultr GPU.

![GPU Stats](assets/images/posts/README/CMyJsoW.jpeg)



# How to build a Llama 2 chatbot

https://blog.streamlit.io/)

Generative AI has been widely adopted, and the development of new, larger, and improved LLMs is advancing rapidly, making it an exciting time for developers.

You may have heard of the recent release of [Llama 2](https://ai.meta.com/llama/?ref=blog.streamlit.io), an open source large language model (LLM) by Meta. This means that you can build on, modify, deploy, and use a local copy of the model, or host it on cloud servers (e.g., [Replicate](https://replicate.com/?ref=blog.streamlit.io)).

While it’s free to download and use, it’s worth noting that self-hosting the Llama 2 model requires a powerful computer with high-end GPUs to perform computations in a timely manner. An alternative is to host the models on a cloud platform like Replicate and use the LLM via API calls. In particular, the three Llama 2 models (`llama-7b-v2-chat`, `llama-13b-v2-chat`, and `llama-70b-v2-chat`) are hosted on Replicate.

In this post, we’ll build a Llama 2 chatbot in Python using Streamlit for the frontend, while the LLM backend is handled through API calls to the Llama 2 model hosted on Replicate. You’ll learn how to:

1. Get a Replicate API token
2. Set up the coding environment
3. Build the app
4. Set the API token
5. Deploy the app



Want to jump right in? Here's the [demo app](https://llama2.streamlit.app/?ref=blog.streamlit.io) and the [GitHub repo](https://github.com/dataprofessor/llama2?ref=blog.streamlit.io).

## What is Llama 2?

Meta released the second version of their open-source Llama language model on July 18, 2023. They’re democratizing access to this model by making it free to the community for both research and commercial use. They also prioritize the transparent and responsible use of AI, as evidenced by their [Responsible Use Guide](https://ai.meta.com/llama/responsible-use-guide?ref=blog.streamlit.io).

Here are the five key features of Llama 2:

1. Llama 2 outperforms other open-source LLMs in benchmarks for reasoning, coding proficiency, and knowledge tests.
2. The model was trained on almost twice the data of version 1, totaling 2 trillion tokens. Additionally, the training included over 1 million new human annotations and fine-tuning for chat completions.
3. The model comes in three sizes, each trained with 7, 13, and 70 billion parameters.
4. Llama 2 supports longer context lengths, up to 4096 tokens.
5. Version 2 has a more permissive license than version 1, allowing for commercial use.

## App overview

Here is a high-level overview of the Llama2 chatbot app:

1. The user provides two inputs: (1) a Replicate API token (if requested) and (2) a prompt input (i.e. ask a question).
2. An API call is made to the Replicate server, where the prompt input is submitted and the resulting LLM-generated response is obtained and displayed in the app.

Let's take a look at the app in action:

<iframe src="https://llama2.streamlit.app/?embed=True" width="600" height="400" scrolling="yes" overflow-y:="" scroll="" style="border: 0px solid; box-sizing: border-box; display: block; vertical-align: middle; --tw-ring-offset-shadow: 0 0 #0000; --tw-ring-shadow: 0 0 #0000; --tw-shadow: var(--tw-shadow-colored); --tw-shadow-colored: 0 20px 25px -5px var(--tw-shadow-color),0 8px 10px -6px var(--tw-shadow-color); aspect-ratio: 16 / 9; border-radius: 1rem; width: 726px; --tw-shadow-color: #f0f2f6; box-shadow: var(--tw-ring-offset-shadow,0 0 #0000),var(--tw-ring-shadow,0 0 #0000),var(--tw-shadow);"></iframe>



1. Go to [https://llama2.streamlit.app/](https://llama2.streamlit.app/?ref=blog.streamlit.io)
2. Enter your Replicate API token if prompted by the app.
3. Enter your message prompt in the chat box, as shown in the screencast below.



[![Llama2-chatbot-screencast_scaling-0.5_fps-15_speed-10.0_duration-0-48](https://blog.streamlit.io/content/images/2023/07/Llama2-chatbot-screencast_scaling-0.5_fps-15_speed-10.0_duration-0-48.gif)](https://blog.streamlit.io/content/images/2023/07/Llama2-chatbot-screencast_scaling-0.5_fps-15_speed-10.0_duration-0-48.gif)



## 1. Get a Replicate API token

Getting your Replicate API token is a simple 3-step process:

1. Go to [https://replicate.com/signin/](https://replicate.com/signin/?ref=blog.streamlit.io).
2. Sign in with your GitHub account.
3. Proceed to the API tokens page and copy your API token.

[![img](https://blog.streamlit.io/content/images/2023/07/Llama2-Replicate-API-token.png)](https://blog.streamlit.io/content/images/2023/07/Llama2-Replicate-API-token.png)

## 2. Set up the coding environment

### Local development

To set up a local coding environment, enter the following command into a command line prompt:

```bash
pip install streamlit replicate
```



NOTE: Make sure to have Python version 3.8 or higher pre-installed.

### Cloud development

To set up a cloud environment, deploy using the Streamlit Community Cloud with the help of the [Streamlit app template](https://github.com/streamlit/app-starter-kit?ref=blog.streamlit.io) (read more [here](https://blog.streamlit.io/streamlit-app-starter-kit-how-to-build-apps-faster/)).

Add a `requirements.txt` file to your GitHub repo and include the following prerequisite libraries:

```bash
streamlit
replicate
```

## 3. Build the app

The Llama 2 chatbot app uses a total of 77 lines of code to build:

```python
import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
```

### Import necessary libraries

First, import the necessary libraries:

- `streamlit` - a low-code web framework used for creating the web frontend.
- `replicate` - an ML model hosting platform that allows interfacing with the model via an API call.
- `os` - the operating system module to load the API key into the environment variable.

```python
import streamlit as st
import replicate
import os
```

### Define the app title

The title of the app displayed on the browser can be specified using the `page_title` parameter, which is defined in the `st.set_page_config()` method:

```python
# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")
```

### Define the web app frontend for accepting the API token

When designing the chatbot app, divide the app elements by placing the app title and text input box for accepting the Replicate API token in the sidebar and the chat input text in the main panel. To do this, place all subsequent statements under `with st.sidebar:`, followed by the following steps:

\1. Define the app title using the `st.title()` method.

\2. Use if-else statements to conditionally display either:

- A success message in a green box that reads `API key already provided!` for the `if` statement.
- A warning message in a yellow box along with a text input box asking for the API token, as none were detected in the Secrets, for the `else` statement.

Use nested if-else statement to detect whether the API key was entered into the text box, and if so, display a success message:

```python
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api
```

### Adjustment of model parameters

In continuation from the above code snippet and inside the same `with st.sidebar:` statement, we're adding the following code block to allow users to select the Llama 2 model variant to use (namely `llama2-7B` or `Llama2-13B`)  as well as adjust model parameters (namely `temperature`, `top_p` and `max_length`).

```python
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
```

### Store, display, and clear chat messages

1. The first code block creates an initial session state to store the LLM generated response as part of the chat message history.
2. The next code block displays messages (via `st.chat_message()`) from the chat history by iterating through the `messages` variable in the session state.
3. The last code block creates a `Clear Chat History` button in the sidebar, allowing users to clear the chat history by leveraging the callback function defined on the preceding line.

```python
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
```

### Create the LLM response generation function

Next, create the `generate_llama2_response()` custom function to generate the LLM’s response. It takes a user prompt as input, builds a dialog string based on the existing chat history, and calls the model using the `replicate.run()` function.

The model returns a generated response:

```python
# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output
```

### Accept prompt input

The chat input box is displayed, allowing the user to enter a prompt. Any prompt entered by the user is added to the session state messages:

```python
# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
```

### Generate a new LLM response

If the last message wasn’t from the assistant, the assistant will generate a new response. While it’s formulating a response, a spinner widget will be displayed. Finally, the assistant's response will be displayed in the chat and added to the session state messages:

```python
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
```

## 4. Set the API token

### Option 1. Set the API token in Secrets

If you want to provide your users with free access to your chatbot, you'll need to cover the costs as your credit card is tied to your account.

To set the API token in the Secrets management on Streamlit Community Cloud, click on the expandable menu at the far right, then click on **Settings**:

[![img](https://blog.streamlit.io/content/images/2023/07/Llama2-Community-Cloud-settings.png)](https://blog.streamlit.io/content/images/2023/07/Llama2-Community-Cloud-settings.png)

To define the `REPLICATE_API_TOKEN` environment variable, click on the **Secrets** tab and paste your Replicate API token:

[![img](https://blog.streamlit.io/content/images/2023/07/Llama2-Community-Cloud-st-secrets.png)](https://blog.streamlit.io/content/images/2023/07/Llama2-Community-Cloud-st-secrets.png)

Once the API token is defined in Secrets, users should be able to use the app without needing to use their own API key:

[![img](https://blog.streamlit.io/content/images/2023/07/Llama2-API-via-st-secrets-1.png)](https://blog.streamlit.io/content/images/2023/07/Llama2-API-via-st-secrets-1.png)

### Option 2. Set the API token in the app

An alternative to setting the API token in Secrets is to prompt users to specify it in the app. This way, users will be notified to provide their own Replicate API token to proceed with using the app:

[![img](https://blog.streamlit.io/content/images/2023/07/Llama2-API-in-app.png)](https://blog.streamlit.io/content/images/2023/07/Llama2-API-in-app.png)

## 5. Deploy the app

Once the app is created, deploy it to the cloud in three steps:

1. Create a GitHub repository for the app.
2. In Streamlit Community Cloud, click on the `New app` button, then choose the repository, branch, and app file.
3. Click `Deploy!` and the app will be live!

## Wrapping up

Congratulations! You’ve learned how to build your own Llama 2 chatbot app using the LLM model hosted on Replicate.

It’s worth noting that the LLM was set to the 7B version and that model parameters (such as `temperature` and `top_p`) were initialized with a set of arbitrary values. This post also includes the Pro version, which allows users to specify the model and parameters. I encourage you to experiment with this setup, adjust these parameters, and explore your own variations. This can be a great opportunity to see how these modifications might affect the LLM-generated response.

For additional ideas and inspiration, check out the [LLM gallery](https://streamlit.io/gallery?category=llms&ref=blog.streamlit.io). If you have any questions, let me know in the comments below or find me on Twitter at [@thedataprof](https://twitter.com/thedataprof?ref=blog.streamlit.io) or on LinkedIn at [Chanin Nantasenamat](https://www.linkedin.com/in/chanin-nantasenamat/?ref=blog.streamlit.io). You can also check out the Streamlit YouTube channel or my personal YouTube channel, [Data Professor](https://youtube.com/dataprofessor?ref=blog.streamlit.io).

Happy chatbot-building! 🦙







To run a GPU on serverless in AWS, you can use **Amazon Elastic Container Service (ECS)**. [Amazon ECS supports workloads that use GPUs when you create clusters with container instances that support GPUs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html)[1](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html). [The EC2 GPU-based container instances that use the p2, p3, g3, g4, g5, and g5g instance types provide access to NVIDIA GPUs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html)[1](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html).

However, running a GPU on AWS Lambda is not possible because you can’t specify the runtime environment for AWS Lambda functions. [The physical machines AWS chooses to put into its Lambda pool will almost certainly not have a GPU](https://stackoverflow.com/questions/52554184/gpu-based-algorithm-on-aws-lambda)[2](https://stackoverflow.com/questions/52554184/gpu-based-algorithm-on-aws-lambda).

If you want to run PyTorch models in AWS Lambda, you can use container images. You can package your code and dependencies as a container image using tools such as the Docker CLI. The maximum container size is 10 GB. After the model for inference is Dockerized, you can upload the image to Amazon Elastic Container Registry (Amazon ECR). [You can then create the Lambda function from the container image stored in Amazon ECR](https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-pytorch-models-in-aws-lambda/)[3](https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-pytorch-models-in-aws-lambda/).

[For more information on running PyTorch models in AWS Lambda using container images, please refer to the provided URL ](https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-pytorch-models-in-aws-lambda/)



Running a container with GPU from a Lambda function is not directly supported by AWS Lambda. However, you can achieve this by combining AWS Lambda with other AWS services like AWS Batch or AWS Elastic Container Service (ECS) that support GPU instances. Here's a high-level overview of the steps involved:

\1. Set up an AWS Batch or ECS cluster with GPU-enabled instances. These services allow you to run containers with GPU support.

\2. Create a Docker container image that includes your application code and any necessary GPU dependencies.

\3. Upload the container image to a container registry like Amazon Elastic Container Registry (ECR) or Docker Hub.

\4. Create an AWS Lambda function that triggers the execution of a job in AWS Batch or ECS.

\5. Configure the Lambda function to pass the necessary parameters and input data to the job.

\6. In the AWS Batch or ECS job definition, specify the GPU-enabled container image and any other required configurations.

\7. When the Lambda function is triggered, it will start the job in AWS Batch or ECS, which will run the container with GPU support.

\8. Monitor the job's progress and retrieve the results as needed.

By combining AWS Lambda with AWS Batch or ECS, you can leverage the serverless nature of Lambda for triggering and managing jobs, while offloading the GPU-intensive tasks to GPU-enabled instances in Batch or ECS.

Please note that this approach involves additional setup and configuration compared to running a GPU-enabled container directly from Lambda. It's recommended to consult the AWS documentation and resources for detailed instructions and best practices specific to your use case.





o set up AWS Elastic Container Service (ECS) with a GPU application, you can follow these steps:

\1. Create a Docker container image that includes your GPU application and any necessary dependencies. Ensure that the image is compatible with the GPU instance type you plan to use.

\2. Upload your container image to a container registry, such as Amazon Elastic Container Registry (ECR), or a third-party registry like Docker Hub.

\3. Set up an ECS cluster by navigating to the AWS Management Console and selecting ECS. Create a new cluster or use an existing one.

\4. Configure your ECS cluster to use GPU-enabled EC2 instances. You can do this by creating a launch template or modifying the instance configuration directly.

\5. Create an ECS task definition that specifies your container image, resource requirements (including GPU), and any other necessary settings. You can define the task definition using the AWS Management Console, AWS CLI, or AWS SDKs.

\6. Create an ECS service using your task definition. This will allow ECS to manage the deployment and scaling of your GPU application across the cluster.

\7. Configure the necessary permissions and roles to allow ECS to access the GPU resources and other required AWS services.

\8. Test your setup by deploying and running your GPU application on the ECS cluster. Monitor the performance and troubleshoot any issues that may arise.

Remember to consider the pricing, availability, and limitations of GPU instances in your chosen region. It's also recommended to consult the AWS documentation and resources for detailed instructions and best practices specific to ECS and GPU applications.

Please note that the availability and specific details of AWS services may change over time, so I recommend referring to the official AWS documentation for the most up-to-date information on setting up ECS with GPU applications.



https://bash-prompt.net/guides/aws-gpu-spot/

https://www.docker.com/blog/deploy-gpu-accelerated-applications-on-amazon-ecs-with-docker-compose/