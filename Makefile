# for ease of use
#

all: iwhipmodel

iwhipmodel: pkg/hipapi/api.go pkg/hipapi/util.go pkg/hipapi/server.go cmd/iwhipmodel/hip.go cmd/iwhipmodel/params.go
	go build cmd/

run: iwhipmodel
	./iwhipmodel
