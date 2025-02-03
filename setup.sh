#!/bin/bash

# Update and upgrade the system
echo "Updating and upgrading system..."
sudo apt update && sudo apt -y full-upgrade 

# Install AI camera firmware
echo "Installing AI camera firmware..."
sudo apt install -y imx500-all 

# Install dependencies
echo "Installing dependencies..."
sudo apt install -y python3-opencv python3-munkres python3-matplotlib python3-skimage

# Install filterpy
echo "Cloning and installing filterpy..."
git clone https://github.com/rlabbe/filterpy.git
cd filterpy
sudo python3 setup.py install
cd ..

# Clone SORT repository
echo "Cloning SORT repository..."
git clone https://github.com/abewley/sort.git
cd sort
cp sort.py ..
cd ..
echo "Setup complete!"