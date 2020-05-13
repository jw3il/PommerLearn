CC := $(CXX)
# We depend on pthread (because of pomlib) and pomlib itself
CFLAGS := -pthread -L ../deps/pomcpp/lib -lpomlib
STD := c++17
SRCDIR := src
SRCEXT := cpp
BUILDDIR := build/src
MAIN_TARGET := ./bin/exec
INCLD := include

DEPS_DIR := ../deps

MAIN_SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))

MAIN_OBJECTS := $(addprefix build/,$(MAIN_SOURCES:.cpp=.o))
# MAIN_OBJS_NOMAIN := $(filter-out $(BUILDDIR)/main.o, $(MAIN_OBJECTS))

# pommerman c++ environment
INC := -I $(DEPS_DIR)/pomcpp/include

all: main

main: $(MAIN_OBJECTS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -std=$(STD) $^ -o $(MAIN_TARGET)
	
build/src/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo "Building main"
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -std=$(STD) -c -o $@ $< $(INC)
	
clean:
	@echo "Cleaning..."; 
	$(RM) -rd $(BUILDDIR) $(MAIN_TARGET)

