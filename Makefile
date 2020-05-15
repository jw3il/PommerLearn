SHELL := /bin/bash

DEPS_DIR := deps
ABS_DEPS_DIR := $(shell pwd)/$(DEPS_DIR)

.DEFAULT_GOAL := all
.PHONY: main deps clean

all: deps main

main:
	echo "> Building agents.."
	@$(MAKE) -s main --directory cpp
	

# Hotfix: Built pomcpp lib contains main.o (error in makefile), we have to remove that.
#         Additionally, we create a link to a modified lib name so that ld can find it easier.
deps:
	@echo "> Building dependencies.."
	
	@echo "-- pomcpp"
	@# Compile C++ pommerman environment
	@$(MAKE) -s lib --directory $(DEPS_DIR)/pomcpp
	
	@echo "-- Modifying pomcpp lib file"
	@ar -dv $(DEPS_DIR)/pomcpp/lib/pomlib.a main.o
	@ln -sf $(ABS_DEPS_DIR)/pomcpp/lib/pomlib.a $(ABS_DEPS_DIR)/pomcpp/lib/libpomlib.a 
    
clean:
	@$(MAKE) -s clean --directory cpp
	@$(RM) -r build
	
	@# clean deps
	@$(MAKE) -s clean --directory $(DEPS_DIR)/pomcpp
	
fresh:
	$(MAKE) clean
	$(MAKE) all

