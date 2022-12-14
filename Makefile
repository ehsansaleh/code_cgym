# the environment name
ENVNAMELIST = InvertedPendulum InvertedDoublePendulum HalfCheetah Swimmer \
							Hopper Walker2d Ant Humanoid HumanoidStandup Reacher Pendulum \
							NLRPendulum leg toy

ENVMJXMLHPPLIST = $(shell for envname in $(ENVNAMELIST); do echo $$envname"/mj_xml.hpp"; done)
ENVSIMIFLIST = $(shell for envname in $(ENVNAMELIST); do echo $$envname"/SimIF"; done)
ENVROLLSOLIST = $(shell for envname in $(ENVNAMELIST); do echo $$envname"/libRollout.so"; done)
ENVROLLOUTLIST = $(shell for envname in $(ENVNAMELIST); do echo $$envname"/Rollout"; done)
CLEANENVSIMIFLIST = $(shell for envname in $(ENVNAMELIST); do echo $$envname"/clean_simif"; done)
CLEANROLLLIST = $(shell for envname in $(ENVNAMELIST); do echo $$envname"/clean_roll"; done)

# the compiler: gcc for C program, define as g++ for C++
CC = module load gcc; g++

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings

CFLAGS  = -O2 --std=c++11 -mavx -pthread -Wl,-rpath,'$$ORIGIN:./lib' -fPIC -g -Wall -faligned-new

# define any directories containing header files other than /usr/include
INCLUDES = -I./include

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS = -L./lib

# This can be used to add the lib directory to the linker's search path
EXPORT_LDLIB = LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(shell realpath ./lib)

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname
#   option, something like (this will link in libmylib.so and libm.so:
SIMIFLIBS = -lmujoco200nogl
MLPLIBS =  -lsleef -lblas
FFTWLIBS =  -lfftw3 -lm

# Define any pre-processor directives in this variable:
SIMIFDEFS = -D xml_type=content_type

SHELL:=/bin/bash

# the build target executable:
SIMIFTARGET = SimIF
MLPTARGET = MlpIF
ROLLTARGET = Rollout
XMLHPPPATH = ./include/mj_xml.hpp

all: $(ENVROLLSOLIST)

################################################
############## Family Definitions ##############
################################################
NLRPendulum_FAMILY = pendulum
Pendulum_FAMILY = pendulum
leg_FAMILY = leg
toy_FAMILY = toy
InvertedPendulum_FAMILY = gym
InvertedDoublePendulum_FAMILY = gym
HalfCheetah_FAMILY = gym
Swimmer_FAMILY = gym
Hopper_FAMILY = gym
Walker2d_FAMILY = gym
Ant_FAMILY = gym
Humanoid_FAMILY = gym
HumanoidStandup_FAMILY = gym
Reacher_FAMILY = gym

################################################
################# SimInterface #################
################################################

$(ENVSIMIFLIST): %/SimIF : %/clean_simif %/mj_xml.hpp $(XMLHPPPATH) ./src/$($*_FAMILY)/$(SIMIFTARGET).cpp
	mkdir -p ./bin/$*
	$(CC) $(CFLAGS) $(INCLUDES) -I ./src/$($*_FAMILY) \
	$(SIMIFDEFS) -D ENVNAME=$* -D xml_file="./src/xml/$*.xml" \
	-D __MAINPROG__=SimIF_CPP -o ./bin/$*/$(SIMIFTARGET) \
	./src/$($*_FAMILY)/$(SIMIFTARGET).cpp $(LFLAGS) $(SIMIFLIBS) $(if $(filter NLRPendulum,$*),$(FFTWLIBS))
	./bin/$*/$(SIMIFTARGET)

$(ENVMJXMLHPPLIST): %/mj_xml.hpp : ./src/xml/%.xml
	@echo "Input XML File -> " ./src/xml/$*.xml
	@echo "#pragma once" > $(XMLHPPPATH)
	@XMLBYTESEXACT=$(shell du -sb ./src/xml/$*.xml | cut -f1); \
	 XMLBYTES=$$(( $$XMLBYTESEXACT + 10 )); \
	 echo "const int xml_bytes = "$$XMLBYTES"; " >> $(XMLHPPPATH); \
	 echo -n "const char xml_content["$$XMLBYTES"] = R\"\"\"(" >> $(XMLHPPPATH)
	@cat ./src/xml/$*.xml >> $(XMLHPPPATH)
	@echo ")\"\"\";" >> $(XMLHPPPATH)

################################################
#################### MLP #######################
################################################

mlp: clean_mlp ./src/$(MLPTARGET).cpp
	$(CC) $(CFLAGS) $(INCLUDES) -I ./src/$($*_FAMILY) -D __MAINPROG__=MlpIF_CPP -o ./bin/$(MLPTARGET) \
	./src/$(MLPTARGET).cpp $(LFLAGS) $(MLPLIBS)
	$(EXPORT_LDLIB) ./bin/$(MLPTARGET)

################################################
################### Rollout ####################
################################################

$(ENVROLLOUTLIST): %/Rollout : %/clean_roll ./src/$(ROLLTARGET).cpp ./src/$(MLPTARGET).cpp
	$(CC) $(CFLAGS) $(INCLUDES) -I ./src/$($*_FAMILY) $(SIMIFDEFS) -D ENVNAME=$* \
	-D __MAINPROG__=Rollout_CPP -D xml_file="./src/xml/$*.xml" \
	-o ./bin/$($*_FAMILY)/$(ROLLTARGET) ./src/$(ROLLTARGET).cpp ./src/$($*_FAMILY)/$(SIMIFTARGET).cpp \
	./src/$(MLPTARGET).cpp $(LFLAGS) $(MLPLIBS) $(SIMIFLIBS) $(if $(filter NLRPendulum,$*),$(FFTWLIBS))
	$(EXPORT_LDLIB) ./bin/$(ROLLTARGET)

$(ENVROLLSOLIST): %/libRollout.so: %/clean_roll %/mj_xml.hpp $(XMLHPPPATH) ./src/$(ROLLTARGET).cpp \
	./src/$(MLPTARGET).cpp
	mkdir -p ./bin/$*
	$(CC) $(CFLAGS) -shared \
	$(INCLUDES) -I ./src/$($*_FAMILY) $(SIMIFDEFS) -D ENVNAME=$* \
	-D __MAINPROG__=Shared_Obj -D xml_file="./src/xml/$*.xml" \
	-o ./bin/$*/lib$(ROLLTARGET).so ./src/$(ROLLTARGET).cpp ./src/$($*_FAMILY)/$(SIMIFTARGET).cpp \
	./src/$(MLPTARGET).cpp $(LFLAGS) $(MLPLIBS) $(SIMIFLIBS) $(if $(filter NLRPendulum,$*),$(FFTWLIBS))
	@echo "-----------------------------------------"

################################################
################# Clearning ####################
################################################

$(CLEANENVSIMIFLIST): %/clean_simif :
	$(RM) ./bin/$*/$(SIMIFTARGET)

$(CLEANROLLLIST): %/clean_roll :
	$(RM) ./bin/$*/lib$(ROLLTARGET).so

clean_mlp:
	$(RM) ./bin/$(MLPTARGET)

clean: $(CLEANENVSIMIFLIST) $(CLEANROLLLIST) clean_mlp
	$(RM) ./bin/core.*
