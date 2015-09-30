CXX := clang++
FLAGS := -lgplib -larmadillo -std=c++11

all: regression

regression: regression.cc
	$(CXX) $(FLAGS) regression.cc -o regression.mio

clean:
	rm -rf *.mio
