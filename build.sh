#! /bin/sh

DEPS_DIR=deps
ABS_DEPS_DIR=$(pwd)/$DEPS_DIR

echo "> Building dependencies.."
echo "-- pomcpp"
# Compile C++ pommerman environment
make -s lib --directory $DEPS_DIR/pomcpp
echo "-- Modifying lib file"
# Hotfix: Built lib contains main.o (error in makefile), we have to remove that
ar -dv $DEPS_DIR/pomcpp/lib/pomlib.a main.o
# Create a link to a modified lib name so that ld can find it easier
ln -sf $ABS_DEPS_DIR/pomcpp/lib/pomlib.a $ABS_DEPS_DIR/pomcpp/lib/libpomlib.a 

echo "> Building agents.."
# Compile C++ agents
make -s main --directory cpp

echo "> Done."

