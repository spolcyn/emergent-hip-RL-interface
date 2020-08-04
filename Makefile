# Copyright (c) 2020, Stephen Polcyn. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

BINARY_NAME=model.exe

PROTO_SRC_DIR=proto
GO_MODEL_DIR=pkg/hipmodel
ALL_PROTO_FILES=$(wildcard $(PROTO_SRC_DIR)/*.proto)
ALL_GO_MODEL_FILES=$(wildcard $(GO_MODEL_DIR)/*.go)

PROTO_NO_PATH=$(subst $(PROTO_SRC_DIR)/,,$(ALL_PROTO_FILES))
ALL_PY_PROTO=$(patsubst %.proto,%_pb2.py,$(PROTO_NO_PATH))
ALL_GO_PROTO=$(addprefix $(GO_MODEL_DIR)/,$(patsubst %.proto, %.pb.go, $(PROTO_NO_PATH)))

all: $(BINARY_NAME)

$(BINARY_NAME): proto cmd/iwhipmodel/main.go $(ALL_GO_MODEL_FILES)
	go build -o "$(BINARY_NAME)" ./cmd/iwhipmodel

proto: $(ALL_PY_PROTO) $(ALL_GO_PROTO)

$(ALL_PY_PROTO): $(ALL_PROTO_FILES)
	protoc -I=$(PROTO_SRC_DIR) --python_out=./ $(ALL_PROTO_FILES)

$(ALL_GO_PROTO): $(ALL_PROTO_FILES)
	protoc -I=$(PROTO_SRC_DIR) --go_out=./ $(ALL_PROTO_FILES)
    
# runs the model with no GUI
# use ./model.exe to run it with a GUI
# technically, using any command-line argument works
run: $(BINARY_NAME)
	./$(BINARY_NAME) nogui

clean: 
	rm -f $(BINARY_NAME) $(ALL_GO_PROTO) $(ALL_PY_PROTO)
