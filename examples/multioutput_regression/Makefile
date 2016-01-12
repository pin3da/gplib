CXX := clang++
FLAGS := -lgplib -larmadillo -std=c++11 -O3

all: mo_full mo_fitc so_fitc

mo_full: mo_full.cc
	$(CXX) $(FLAGS) mo_full.cc -o mo_full.mio

mo_fitc: mo_fitc.cc
	$(CXX) $(FLAGS) mo_fitc.cc -o mo_fitc.mio

so_fitc: so_fitc.cc
	$(CXX) $(FLAGS) so_fitc.cc -o so_fitc.mio

clean:
	rm -rf *.mio
	rm -rf *.png
