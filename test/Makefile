CC=g++
CFLAGS=-std=c++11 -O3 -Wall
LDFLAGS=-larmadillo -lgplib

all:  kernel_gradient

kernel_gradient: kernel_gradient.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) kernel_gradient.cpp -o kernel_gradient.test

clean:
	rm kernel_gradient.test
