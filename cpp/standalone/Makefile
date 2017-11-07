# TF_ROOT=../tf_dist

BIN_SRCS=facenet_tf.cpp
COMM_SRCS+=utils.cpp

# BIN_SRCS=test.cpp  camera.cpp facenet_tf.cpp
# COMM_SRCS+=tensorflow_mtcnn.cpp comm_lib.cpp  utils.cpp

#
# opencv settings
#

OPENCV_LDFLAGS=$(shell pkg-config --libs opencv)
OPENCV_CFLAGS=$(shell pkg-config --cflags opencv)


CXXFLAGS += $(OPENCV_CFLAGS) 
LIBS += $(OPENCV_LDFLAGS)

#
# tensorflow ettings
#


TF_INCS += -I/usr/include
CXXFLAGS += $(TF_INCS) 
LIBS+=-Wl,-rpath,/usr/lib -ltensorflow
# LIBS+=-Wl,-rpath -ltensorflow


CXX := g++
CXXFLAGS+= -Wall  -ggdb -std=c++14


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

clean:
	rm -f $(BIN_EXES) *.o

.PHONY : all clean

