## Provide your own options or compiler here.
CC = gcc
#CC = icc
#CFLAGS +=		-g
CFLAGS += 	-O3 
CFLAGS += 	-Wl,-no-as-needed 
CFLAGS += 	-malign-double 
CFLAGS += 	-msse3
CFLAGS +=		-funroll-loops
CFLAGS += 	-ftree-vectorize 
#CFLAGS += 	-ffast-math 
CFLAGS += 	-ftree-vectorizer-verbose=2 -msse2
CFLAGS += 	-march=native #-fprefetch-loop-arrays


## You shouldn't need to edit anything past this point.

APP = matmul
SRCS = driver.c matmul.c utils.c

## Check for the PAPI header file. We assume that if we have the
## header, we have the library as well.
ifneq ($(shell ls /kozyrakis/tools/papi/include/papi.h 2> /dev/null),)
  LDFLAGS += -lpapi -L/kozyrakis/tools/papi/lib -Wl,-rpath,/kozyrakis/tools/papi/lib,--enable-new-dtags
  CFLAGS += -DPAPI -I/kozyrakis/tools/papi/include
endif

ifneq ($(shell ls /usr/lib/libblas.so 2> /dev/null),)
  LDFLAGS += -lblas
  CFLAGS += -DBLAS
endif

all: $(APP)

$(APP): $(SRCS:.c=.o)
	$(LINK.c) $^ $(LOADLIBES) $(LDLIBS) -o $@

clean:
	rm -f *.o *~ $(APP)

# Some generate dependencies.
%.o: %.c utils.h Makefile
	$(COMPILE.c) $(OUTPUT_OPTION) $<
