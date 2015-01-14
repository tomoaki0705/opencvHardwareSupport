CC = g++
LD = g++
LDFLAGS=`pkg-config --libs opencv`
CFLAGS =`pkg-config --cflags opencv`
HWSUPPORT = hwSupport

all: $(HWSUPPORT)
.PHONY: clean

clean:
	@rm -rf main.o $(HWSUPPORT)

$(HWSUPPORT): main.o
	@echo "LD    : " $@
	@$(LD) -o $@ $< $(LDFLAGS)

main.o: main.cpp
	@echo "CC    : " $@
	@$(CC) -o $@ -c $< $(CFLAGS)
