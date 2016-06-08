CXX = mpicxx 
CXXFLAGS = --std=c++11 -lm -fopenmp -g -O0

SRCS = $(wildcard *cpp)

TARGETS = $(SRCS:.cpp=)
.PHONY: clean help

all: $(TARGETS)

%: %.cpp
	 $(CXX) $(CXXFLAGS) -o $@ $^

clean:
	@echo "remove $(TARGETS)"
	rm -f $(TARGETS)

help:
	@echo "make"
	@echo "make clean"
	@echo "make help"
