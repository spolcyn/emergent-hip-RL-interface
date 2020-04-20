# for ease of use
#

all: iwhipmodel

iwhipmodel: api.go util.go server.go hip.go
	go build

run: iwhipmodel
	./iwhipmodel
