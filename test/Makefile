CC=g++
CFLAGS=-std=c++11 -O3 -Wall
LDFLAGS=-larmadillo -lgplib

all:  kernel_gradient

kernel_gradient: kernel_gradient.cc
	$(CC) $(CFLAGS) $(LDFLAGS) kernel_gradient.cc -o kernel_gradient.test

clean:
	rm kernel_gradient.test
