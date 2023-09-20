## Install the CUDA Toolkitm

To run Llama 2 models with lower precision settings, the CUDA toolkit is essential. Install the toolkit to install the libraries needed to write and compile GPU-accelerated applications using CUDA as described in the steps below.

```
sudo apt update
```

```
sudo apt install build-essential -y
```



1. Download the latest CUDA toolkit version.

   ```
   $ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   ```

2. Initialize the CUDA toolkit installation.

   ```
   $ sudo sh cuda_11.8.0_520.61.05_linux.run
   ```

   When prompted, read the CUDA terms and conditions. Enter `accept` to agree to the toolkit license. Then, in the installation prompt, press SPACE to deselect all any provided options, and only keep the CUDA toolkit selected. Using arrow keys, scroll to the `Install` option and press ENTER to start the installation process.

3. Using `echo`, append the following configurations at the end of the `~/.bashrc` file.

   ```
   $ echo " export PATH=$PATH:/usr/local/cuda-11.8/bin
   
            export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 " >> ~/.bashrc
   ```

   The above configuration lines declare the environment variable configurations that allow your system to use the CUDA toolkit and its libraries.

4. Using a text editor such as `Vim`, edit the `/etc/ld.so.conf/cuda-11-8.conf` file.

   ```
   $ sudo vim /etc/ld.so.conf.d/cuda-11-8.conf
   ```

5. Add the following configuration at the beginning of the file.

   ```
   /usr/local/cuda-11.8/lib64
   ```

   Save and close the file.

6. To save the configuration, end your SSH session.

   ```
   $ exit
   ```

7. Start a new SSH session.

   ```
   $ ssh example-user@SERVER-IP
   ```

8. Run the following `ldconfig` command to update the linker cache, and refresh information about shared libraries on your server.

   ```
   $ sudo ldconfig
   ```

## 