EXEC = \
    tests/test-matrix \
    tests/test-stopwatch \
	tests/stat-matrix

GIT_HOOKS := .git/hooks/applied
OUT ?= .build
.PHONY: all
all: $(GIT_HOOKS) $(OUT) $(EXEC)

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

CC ?= gcc
CFLAGS = -Wall -std=gnu99 -g -msse4.1 -mavx -mavx2 -O0 -I.
LDFLAGS = -lpthread

OBJS := \
	stopwatch.o \
	matrix_naive.o \
	matrix_sse.o \
	matrix_avx.o

deps := $(OBJS:%.o=%.o.d)
OBJS := $(addprefix $(OUT)/,$(OBJS))
deps := $(addprefix $(OUT)/,$(deps))

tests/test-%: $(OBJS) tests/test-%.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tests/stat-%: $(OBJS) tests/stat-%.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OUT)/%.o: %.c $(OUT)
	$(CC) $(CFLAGS) -c -o $@ -MMD -MF $@.d $<

$(OUT):
	@mkdir -p $@

data.csv: tests/stat-matrix
	./tests/stat-matrix

plot: data.csv
	gnuplot scripts/runtime.gp

check: $(EXEC)
	@for test in $^ ; \
	do \
		echo "Execute $$test..." ; $$test && echo "OK!\n" ; \
	done

clean:
	$(RM) $(EXEC) $(OBJS) $(deps)
	@rm -rf $(OUT) *.csv *.png

-include $(deps)
