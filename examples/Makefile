CXX := clang++
FLAGS := -lgplib -larmadillo -std=c++11

all: regression

regression: regression.cc
	$(CXX) regression.cc -o regression $(FLAGS)
clean:
	rm -rf regression
