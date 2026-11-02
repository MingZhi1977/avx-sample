CC ?= gcc
CFLAGS ?= -O2 -mavx2 -mfma -Wall -Wextra
TARGET = avx2_demo
SRC = avx2_demo.c

.PHONY: all run clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

help:
	@echo "可用指令:"
	@echo "  make        - 編譯 AVX2 示範程式"
	@echo "  make run    - 編譯並直接執行示範"
	@echo "  make clean  - 清理編譯產物"
