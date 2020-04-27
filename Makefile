# for ease of use
#

all: iwhipmodel

iwhipmodel: pkg/hipmodel/api.go pkg/hipmodel/util.go pkg/hipmodel/server.go pkg/hipmodel/hip.go pkg/hipmodel/params.go pkg/hipmodel/hip_utils.go cmd/iwhipmodel/main.go
	go build ./cmd/iwhipmodel

run: iwhipmodel
	./iwhipmodel
