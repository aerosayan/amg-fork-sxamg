
default: lib  amg amg-low-level cg-amg poisson

lib:
	@(cd ../; make)

include ../Makefile.inc

amg.o: amg.c ../src/lib$(SXAMGLIB).a
amg-low-level.o: amg-low-level.c ../src/lib$(SXAMGLIB).a
cg-amg.o: cg-amg.c ../src/lib$(SXAMGLIB).a
poisson.o: poisson.c ../src/lib$(SXAMGLIB).a

clean:
	rm -f *.o amg amg-low-level cg-amg poisson
