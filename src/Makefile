VER_MAJOR := 1
VER_MINOR := 0
VERSION := $(VER_MAJOR).$(VER_MINOR)
NAME := gplib
LIBNAME := lib$(NAME)
OBJ_FILES := basic.o mvgauss.o gpreg.o kernels.o
HEADERS := $(shell find . -name "*.h")
INSTALL_PATH := /usr/local/lib
INCLUDE_PATH := /usr/local/include
INCFOLDER := $(INCLUDE_PATH)/$(NAME)_folder
FLAGS := -fPIC -Wall -std=c++11 -g

all: $(LIBNAME).so

test: $(HEADERS)
	for i in $(HEADERS); do echo "mkdir -p $$(dirname $(INCLUDE_PATH)/$(NAME)/$$i)" ; done

$(LIBNAME).so: $(OBJ_FILES)
	g++ -shared -Wl,-soname,$(LIBNAME).so.$(VER_MAJOR) -o $(LIBNAME).so $(OBJ_FILES)

%.o : %.cpp
	g++ -c $(FLAGS) -o $@ $<

install: $(LIBNAME).so
	rm -f $(INSTALL_PATH)/$(LIBNAME).so*
	rm -f -r $(INCFOLDER)
	rm -f $(INCLUDE_PATH)/gplib
	cp $(LIBNAME).so $(INSTALL_PATH)/$(LIBNAME).so.$(VERSION)
	ln -s $(INSTALL_PATH)/$(LIBNAME).so.$(VERSION) $(INSTALL_PATH)/$(LIBNAME).so.$(VER_MAJOR)
	ln -s $(INSTALL_PATH)/$(LIBNAME).so.$(VERSION) $(INSTALL_PATH)/$(LIBNAME).so
	mkdir $(INCFOLDER)
	for i in $(HEADERS); do mkdir -p "$$(dirname $(INCFOLDER)/$$i)" ; cp $$i $(INCFOLDER)/$$i; done
	cp ../gplib $(INCLUDE_PATH)
	ldconfig

uninstall:
	rm $(INSTALL_PATH)/$(LIBNAME).so*
	rm -r $(INCLUDE_PATH)/$(NAME)
	rm $(INCLUDE_PATH)/gplib
	ldconfig

clean:
	rm *.o *.so
