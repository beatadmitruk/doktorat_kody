CXX = icc
CXXFLAGS = -O2 -qopenmp -fprotect-parens -std=c99

EXE_NAME_D = ./build/dvsumk
NAME_D_SOURCES = nowy_dkh.c

EXE_NAME_S = ./build/vsumgm
NAME_S_SOURCES = xsmain.c


all: build_s

build_d:
        $(CXX) $(CXXFLAGS) -o $(EXE_NAME_D) $(NAME_D_SOURCES)

build_s:
        $(CXX) $(CXXFLAGS) -o $(EXE_NAME_S) $(NAME_S_SOURCES)

clean:
        rm -f $(EXE_NAME_D) $(EXE_NAME_S)
