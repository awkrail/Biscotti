CC := g++

dump_dct: dump_dct.cc
	$(CC) dump_dct.cc -o bin/dump_dct -ljpeg

