## Build and Run the code
### Build
- Build from command line
- Simple build: g++ opencv_cuda.cpp -o detect
- Include opencv lib: g++ opencv_cuda.cpp -o detect -I/usr/include/opencv (find openCV library path using command $ pkg-config --cflags --libs opencv4
- Include .h files:
### Build using CMake
- mkdir -p build
- cd build
- cmake ..
- make
- ./detect

## Results
- Webcam single frame average read and display time using CPU: 35 mSec 
