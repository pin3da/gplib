CXX := clang++
FLAGS := -lgplib -larmadillo -std=c++11

all: regression mo_gen

regression: regression.cc
	$(CXX) $(FLAGS) regression.cc -o regression.mio

mo_gen:  mo_generator.cc
	$(CXX) $(FLAGS) mo_generator.cc -o mo_gen.mio

clean:
	rm -rf *.mio
