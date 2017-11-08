TF_ROOT=./tf140
OPENCV_ROOT=./opencv330

#BIN_SRCS=facenet_tf.cpp
#COMM_SRCS+=utils.cpp

BIN_SRCS=test.cpp  facenet_tf.cpp # camera.cpp
JNI_SRCS=jni_types.hpp jni_common.cpp
COMM_SRCS+=tensorflow_mtcnn.cpp comm_lib.cpp  utils.cpp tf_mtcnn.cpp opencv_utils.cpp $(JNI_SRCS)

OPENCV_LDFLAGS=-lopencv_core -lopencv_imgcodecs -lopencv_imgproc
CXXFLAGS += $(OPENCV_CFLAGS)
LIBS += $(OPENCV_LDFLAGS)

#CXXFLAGS += -I/usr/include -I$(OPENCV_ROOT)/include -I$(TF_ROOT)/include
CXXFLAGS += -I$(OPENCV_ROOT)/include -I$(TF_ROOT)/include
LIBS+=-Wl,-rpath,-L$(OPENCV_ROOT)/lib,-L$(TF_ROOT)/lib,-ltensorflow
# LIBS+=-Wl,-rpath -ltensorflow

CXX := g++
CXXFLAGS+= -Wall -Wextra -ggdb -std=c++14
# LDFLAGS = -static
CXXFLAGS+=-fPIC -g
#LDFLAGS=-shared

COMM_OBJS=$(COMM_SRCS:.cpp=.o)
BIN_OBJS=$(BIN_SRCS:.cpp=.o)
BIN_EXES=$(BIN_SRCS:.cpp=)

default : $(BIN_EXES)

$(BIN_EXES) : $(COMM_OBJS)

$(BIN_EXES):%:%.o

%:%.o
	$(CXX) $< -o $@ $(LDFLAGS) $(COMM_OBJS) $(LIBS)

%.o : %.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

release:
	echo "relaase..."

clean:
	rm -f $(BIN_EXES) *.o

.PHONY : all clean

