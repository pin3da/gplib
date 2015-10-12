CXX := clang++
FLAGS := -lgplib -larmadillo -std=c++11

all: mo_full mo_fitc

mo_full: mo_full.cc
	$(CXX) $(FLAGS) mo_full.cc -o mo_full.mio


mo_fitc: mo_fitc.cc
	$(CXX) $(FLAGS) mo_fitc.cc -o mo_fitc.mio

mo_gen:  mo_generator.cc
	$(CXX) $(FLAGS) mo_generator.cc -o mo_gen.mio

clean:
	rm -rf *.mio
