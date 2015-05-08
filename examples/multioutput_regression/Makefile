CXX := clang++
FLAGS := -lgplib -larmadillo -std=c++11

all: multioutput

multioutput: multioutput.cc
	$(CXX) $(FLAGS) multioutput.cc -o multioutput.mio

mo_gen:  mo_generator.cc
	$(CXX) $(FLAGS) mo_generator.cc -o mo_gen.mio

clean:
	rm -rf *.mio
