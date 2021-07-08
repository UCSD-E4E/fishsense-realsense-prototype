CXX = g++
TARGET = align-p2p
OPENCV =`pkg-config opencv --cflags --libs`
RS = -lrealsense2

make:

	$(CXX) -std=c++11 $(TARGET).cpp $(RS) $(OPENCV) -o $(TARGET)
