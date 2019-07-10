# DGXOS-HPC Install Instructions

Install instructions for DGX OS HPC.

<br>

---
**WARNING:** Once this package has been installed, the changes can only be removed with a system reimage. Uninstalling/removing the package with `apt-get` will not revert these changes. Ensure your system is properly backed up before continuing.

**WARNING:** By installing this package, all the Spectre/Meltdown/MDS and other mitigations are being disabled. This configuration is only expected to run in a trusted environment.

---
<br>

## Requirement
DGX-1 with 16Gb V100 GPUs or DGX2 or DGX2H
  

## Prerequisites
### Base OS
These instructions require that DGX Base OS 4.1.0 is installed. To check which OS version is currently installed, run the following command:  

`grep VERSION /etc/dgx-release`  
```bash
DGX_SWBUILD_VERSION="4.1.0"
```

Before beginning the installation, please ensure you are running Base OS 4.1.0. If not, follow the [NVIDIA DGX OS SERVER VERSION 4.1.0 Release Notes and Update Guide](https://docs.nvidia.com/dgx/pdf/DGX-OS-server-4.1.0-relnotes-update-guide.pdf) before continuing.

### Kernel and Security Updates
The latest kernel and other security patches must be installed before installing the DGX OS HPC packages. Run the following commands to ensure your machine is up to date: 

`sudo apt-get update`  
`sudo apt-get dist-upgrade`  
`sudo apt-get autoremove`

Notes: 
 * Depending on the DGX configuration, it's possible the command `sudo apt-get dist-upgrade` will display one or more user prompts asking whether to keep a local config file or take the package maintainer's version. 
 * If you haven't manually changed any of these files, it's recommended to take the package maintainer's version.
 * To automatically use the maintainer's version, the `-y` flag can be specified. i.e. `sudo apt-get -y dist-upgrade`
   * This should only be used when the user *knows* none of the local files have been manually updated, such as after a system reimage.

## Installation

1. Enable the dgxos-hpc distro:  

   `sudo apt-get install dgxos-hpc-repo`  
   `sudo apt-get update`
2. Install the custom performance configuration:  

   `sudo apt-get install dgxos-hpc-configs`  
   
   * This will purge the NVSM and fail2ban tools, and their dependencies. 
   * If prompted to keep the currently installed version of a config file or take the package maintainer's version, then take the package maintainer's version of a config file.
3. Restart the server:  

   `sudo reboot`
4. Verify the installation was successful:  

   `tail /sys/devices/system/cpu/vulnerabilities/* | grep -C 1 "Vulnerable"`  

   Expected Output:
```bash
==> /sys/devices/system/cpu/vulnerabilities/mds <==
Vulnerable; SMT vulnerable

==> /sys/devices/system/cpu/vulnerabilities/meltdown <==
Vulnerable

--
==> /sys/devices/system/cpu/vulnerabilities/spectre_v2 <==
Vulnerable, IBPB: disabled, STIBP: disabled
```

## Notes

1. This will purge the NVSM and fail2ban tools, and their dependencies.
2. There are no upgrade/downgrade paths from the `dgxos-hpc` distro.
