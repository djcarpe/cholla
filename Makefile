# Builds the Cholla NIF. The Needle engine is not linked here: the NIF dlopens
# the prebuilt, self-contained libneedle at runtime (see Cholla.Engine).
PRIV_DIR = $(MIX_APP_PATH)/priv
NIF = $(PRIV_DIR)/cholla_nif.so

ERTS_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~ts/erts-~ts/include/", [code:root_dir(), erlang:system_info(version)]), halt().')

CFLAGS ?= -O2 -Wall -Wextra -Wno-unused-parameter
CFLAGS += -std=c11 -fPIC -fvisibility=hidden -I$(ERTS_INCLUDE_DIR) -Ic_src

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -dynamiclib -undefined dynamic_lookup
else
	LDFLAGS += -shared
	LDLIBS += -ldl
endif

all: $(NIF)

$(NIF): c_src/cholla_nif.c c_src/needle.h
	@mkdir -p $(PRIV_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ c_src/cholla_nif.c $(LDLIBS)

clean:
	rm -f $(NIF)

.PHONY: all clean
