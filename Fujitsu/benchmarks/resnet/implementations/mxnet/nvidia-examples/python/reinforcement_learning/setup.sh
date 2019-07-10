#!/bin/bash

# Set up ALE
echo "Setting up Arcade Learning Environment"
apt-get install -y libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
mkdir build && cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
make -j 4
make install
cd ..
pip install .
cd ..
rm -rf Arcade-Learning-Environment

# Get the Atari Breakout ROM
wget https://atariage.com/2600/roms/Breakout.zip
unzip Breakout.zip
rm Breakout.zip 
