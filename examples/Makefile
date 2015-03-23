CXX := clang++
FLAGS := -lgplib -larmadillo -std=c++11

all: regression

regression: regression.cc
	$(CXX) regression.cc -o regression.mio $(FLAGS)
clean:
	rm -rf regression
