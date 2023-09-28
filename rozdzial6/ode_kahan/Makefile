CXX = icc
CXXFLAGS = -O2 -qopenmp -fprotect-parens -std=c99

EXE_NAME_D = ./bulid/dbvp
NAME_D_SOURCES = dbvp3.c

EXE_NAME_S = ./bulid/fbvp
NAME_S_SOURCES = fbvp3.c

all: build_d build_s

build_d:
        $(CXX) $(CXXFLAGS) -o $(EXE_NAME_D) $(NAME_D_SOURCES)

build_s:
        $(CXX) $(CXXFLAGS) -o $(EXE_NAME_S) $(NAME_S_SOURCES)

clean:
        rm -f $(EXE_NAME_D) $(EXE_NAME_S)
